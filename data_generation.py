import numpy as np
# Legend : int/floats of [x,y] continuous range from x to y
#       : int/floats of (x,y,z) discrete x,y or z
#       : string (x,y,z) discrete x,y or z
from data.data_utils import DataUtils
from processing.data2image import bodyToImg, headshapeToImg
import math
import pprint
import os
import json
from tqdm import tqdm
import cv2 as cv

class DataGen:
    def __init__(self):
        # models available in DeepFace
        #       "VGG-Face" - Downloaded
        #       "Facenet512" - Downloaded
        #       "OpenFace" 
        #       "DeepFace" - Downloaded
        #       "DeepID" 
        #       "ArcFace" 
        #       "Dlib" 
        #       "SFace"

        self.dataGen = DataUtils(verb=False)
        self.directions = ["left", "right", "up", "down"]
        self.model = "VGG-Face"
        self.dataJsonFileName = "data.json"
        self.gtNPYFileName = "GT.npy"
        self.embFileName = "embeddings.npy"
        self.dataPath = "./data"
         
    def arrayShapes(self):
        assert os.path.exists(os.path.join(self.dataPath,"ground_truth",self.gtNPYFileName)), "Ground truth array file needs to be generated first"
        assert os.path.exists(os.path.join(self.dataPath,"embeddings",self.embFileName)), "Embeddings array file needs to be created first"
        
        gt = np.load(os.path.join("./data","ground_truth","GT.npy"))
        emb = np.load(os.path.join("./data","embeddings","embeddings.npy"))
        return {"ground_truth": gt.shape,"embeddings": emb.shape}
        
    def inputData2Dict(self, x):
        lengths = [4, 3, 13, 1, 4, 11, 18, 12, 12, 13, 12, 12, 11, 2, 2]
        assert len(x) == sum(lengths)
        ind = 0
        features_categories = {"body": None, "headshape": None, "eyes": None, "eye_colour": None,
                               "eyebrows": None, "forehead": None, "nose": None, "ears": None,
                               "cheeks": None, "mouth": None, "jaw": None, "chin": None,
                               "neck": None, "facial_forms1": None, "facial_forms2": None}
        # Populate the dict
        for i, cat in enumerate(features_categories.keys()):
            features_categories[cat] = x[ind:ind+lengths[i]]
            ind = ind + lengths[i]

        return features_categories

    def encodeInput(self, x):
        """Function which transforms the input data to the required format for training

        Args:
            x (NumPy array): The input feature as a numpy array

        Returns:
            output (NumPy array): the output vector
        """

        def _bodyTransform(x):
            radCoeff = np.pi/180

            body = x["body"]
            r = float(body[0])  # it is already in [0,1] so its fine
            theta = float(body[1])
            gender = str(body[2])
            skintone = int(body[3])  # 9 total options -> one hot encode

            # Transform the degrees to x and y components -> [-1,1]
            xcos = np.cos(theta*radCoeff)
            ysin = np.sin(theta*radCoeff)

            # Encode gender
            gender_enc = 0 if gender=="male" else 1
            
            # One hot encode
            skintone_enc = np.zeros(9)  # total number of options
            skintone_enc[skintone-1] = 1
            
            # Combine all and replace
            bodyTransofrmed = np.append(
                np.array([r, xcos, ysin,gender_enc]), skintone_enc)
            x["body"] = bodyTransofrmed
            return x

        def _headshapeTransform(x):
            headshape = x["headshape"]

            headshapePreset = int(headshape[0])
            refineDirection = str(headshape[1])
            intensity = float(headshape[2])

            # One hot encode preset
            headshapePreset_enc = np.zeros(9)
            headshapePreset_enc[headshapePreset-1] = 1

            # Encode direction into a vector and scale by intensity
            refineVector = np.array([0, 0])
            if refineDirection == "left":
                refineVector = np.array([-1, 0])
            elif refineDirection == "right":
                refineVector = np.array([1, 0])
            elif refineDirection == "up":
                refineVector = np.array([0, 1])
            elif refineDirection == "down":
                refineVector = np.array([0, -1])
            refineVector = refineVector*intensity

            # Combine all and replace
            headshapeTransformed = np.append(headshapePreset_enc, refineVector)
            x["headshape"] = headshapeTransformed
            return x

        def _eyeColourTransform(x):
            eyeColour = int(x["eye_colour"][0])

            # One hot encode eye colour
            eyeColour_enc = np.zeros(12)
            eyeColour_enc[eyeColour-1] = 1

            # Replace original
            x["eye_colour"] = eyeColour_enc
            return x

        def _eyebrowsTransform(x):
            eyebrows = x["eyebrows"].astype(np.float32)

            preset = int(eyebrows[0])
            sliders = eyebrows[1:]

            # One hot encode the eyebrow preset
            preset_enc = np.zeros(10)
            preset_enc[preset-1] = 1

            # Combine and replace
            eyebrowsTransformed = np.append(preset_enc, sliders)
            x["eyebrows"] = eyebrowsTransformed

            return x

        def _facialFormsTransform(x):
            FF1 = x["facial_forms1"].astype(np.float32)
            FF2 = x["facial_forms2"].astype(np.float32)

            FF1_preset = int(FF1[0])
            FF2_preset = int(FF2[0])
            FF1_intensity = FF1[1]
            FF2_intensity = FF2[1]

            # One hot encode the facial form presets
            FF1_preset_enc = np.zeros(16)
            FF2_preset_enc = np.zeros(16)
            FF1_preset_enc[FF1_preset-1] = 1
            FF2_preset_enc[FF2_preset-1] = 1

            # Combine and Replace
            FF1Transformed = np.append(FF1_preset_enc, FF1_intensity)
            FF2Transformed = np.append(FF2_preset_enc, FF2_intensity)
            x["facial_forms1"] = FF1Transformed
            x["facial_forms2"] = FF2Transformed

            return x

        def _typeTransform(x):
            for key in x.keys():
                x[key] = x[key].astype(np.float32)
            return x

        def _toVector(x):
            featureVector = np.array([])
            for value in x.values():
                featureVector = np.append(featureVector, value)
            return featureVector

        # The transfroms require a dict as an input so first the data is turned into a dict
        inputDict = self.inputData2Dict(x)
        # Apply transforms
        transforms = [_bodyTransform, _headshapeTransform,
                      _eyeColourTransform, _eyebrowsTransform, _facialFormsTransform, _typeTransform, _toVector]
        for tr in transforms:
            encodedInput = tr(inputDict)

        return np.round(encodedInput, 2)

    def generateDataExample(self):
        """ Function which generates random data points (in game settings)

        Returns:
            out (np.array) : Transformed output vector, encoded for ML use
        """

        featureList = []
        # Body - radius, angle, skintone
        # R: [0, 1] ; Theta: (0, ..., 360) ; skintone (1, ..., 9)
        r = self.dataGen.getRandomFloatVector(1, 0, 1)

        # if the radius is 0 the angle is set to 0
        # the rationale is that we want to reduce variation in the data which doesnt result in information gain
        if r[0] == 0:
            theta = np.array([0])
        else:
            theta = self.dataGen.getRandomIntVector(0, 360)

        bodySkintone = self.dataGen.getRandomIntVector(1, 9)
        bodyGender = self.dataGen.getRandomStringVector(["male","female"])
        body = np.append(r, (theta, bodyGender, bodySkintone))
        assert len(body) == 4
        featureList.append(body)

        # Head shape - preset, refine direction, intensity
        # preset: (1, ..., 9) ; refine_dir: (left,right,up,down) ; intensity: [0,1]
        # Headshape preset depends on the character's gender
        if bodyGender == "male":
            headshapePreset = self.dataGen.getRandomIntVector(1, 9)
        else:
            headshapePreset = self.dataGen.getRandomIntVector(1, 8)
        headshapeRefineDir = self.dataGen.getRandomStringVector(self.directions)
        headshapeRefineIntensity = self.dataGen.getRandomFloatVector(1, 0, 1)
        headshape = np.append(headshapePreset, (headshapeRefineDir, headshapeRefineIntensity))
        assert len(headshape) == 3
        featureList.append(headshape)

        # Eyes - 4 sliders + 9 mixin sliders\
        # sliders 1 to 4 : [-1,1] ; mix_sliders 1 to 9: [0, 1] | sum(mix_sliders 1 to 9) !> 1.0
        eyes = np.append(self.dataGen.getRandomFloatVector(4, -1, 1),
                         self.dataGen.getRandomMixVector("EYES"))
        assert len(eyes) == 13
        featureList.append(eyes)

        # Eye Colour - preset
        # preset: (1, ..., 12)
        eye_colour = self.dataGen.getRandomIntVector(1, 12)
        assert len(eye_colour) == 1
        featureList.append(eye_colour)

        # Eyebrows - preset, 3 sliders
        # preset: (1, ..., 10) ; sliders 1 to 3: [-1,1]
        eyebrows = np.append(self.dataGen.getRandomIntVector(1, 10),
                             self.dataGen.getRandomFloatVector(3, -1, 1))
        assert len(eyebrows) == 4
        featureList.append(eyebrows)

        # Forehead - 2 sliders, 9 mixins
        # sliders 1 and 2: [-1,1] ; mix_sliders 1 to 9: [0,1] | sum(mix_sliders 1 to 9) !> 1.0
        forehead = np.append(self.dataGen.getRandomFloatVector(2, -1, 1),
                             self.dataGen.getRandomMixVector("FOREHEAD"))
        assert len(forehead) == 11
        featureList.append(forehead)

        # Nose - 9 sliders, 9 mixins
        # sliders 1 to 9: [-1,1] ; mix_sliders 1 to 9: [0,1] | sum(mix_sliders 1 to 9) !> 1.0
        nose = np.append(self.dataGen.getRandomFloatVector(9, -1, 1),
                         self.dataGen.getRandomMixVector("NOSE"))
        assert len(nose) == 18
        featureList.append(nose)

        # Ears - 3 sliders, 9 mixins
        # sliders 1 to 3: [-1,1] ; mix_sliders 1 to 9: [0,1] | sum(mix_sliders 1 to 9) !> 1.0
        ears = np.append(self.dataGen.getRandomFloatVector(3, -1, 1),
                         self.dataGen.getRandomMixVector("EARS"))
        assert len(ears) == 12
        featureList.append(ears)

        # Cheeks - 3 sliders, 9 mixins
        # sliders 1 to 3: [-1,1] ; mix_sliders 1 to 9: [0,1] | sum(mix_sliders 1 to 9) !> 1.0
        cheeks = np.append(self.dataGen.getRandomFloatVector(3, -1, 1),
                           self.dataGen.getRandomMixVector("CHEEKS"))
        assert len(cheeks) == 12
        featureList.append(cheeks)

        # Mouth - 4 sliders, 9 mixins
        # sliders 1 to 4: [-1,1] ; mix_sliders 1 to 9: [0,1] | sum(mix_sliders 1 to 9) !> 1.0
        mouth = np.append(self.dataGen.getRandomFloatVector(4, -1, 1),
                          self.dataGen.getRandomMixVector("MOUTH"))
        assert len(mouth) == 13
        featureList.append(mouth)

        # Jaw - 3 sliders, 9 mixins
        # sliders 1 to 3: [-1,1] ; mix_sliders 1 to 9: [0,1] | sum(mix_sliders 1 to 9) !> 1.0
        jaw = np.append(self.dataGen.getRandomFloatVector(3, -1, 1),
                        self.dataGen.getRandomMixVector("JAW"))
        assert len(jaw) == 12
        featureList.append(jaw)

        # Chin - 3 sliders, 9 mixins
        # sliders 1 to 3: [-1,1] ; mix_sliders 1 to 9: [0,1] | sum(mix_sliders 1 to 9) !> 1.0
        chin = np.append(self.dataGen.getRandomFloatVector(3, -1, 1),
                         self.dataGen.getRandomMixVector("CHIN"))
        assert len(chin) == 12
        featureList.append(chin)

        # Neck - 2 sliders, 9 mixins
        # sliders 1 to 2: [-1,1] ; mix_sliders 1 to 9: [0,1] | sum(mix_sliders 1 to 9) !> 1.0
        neck = np.append(self.dataGen.getRandomFloatVector(2, -1, 1),
                         self.dataGen.getRandomMixVector("NECK"))
        assert len(neck) == 11
        featureList.append(neck)

        # Facial forms - preset, intensity
        # preset: (1, ..., 16) ; intensity: [0,1]
        # Facial forms is available twice in-game
        facial_forms1 = np.append(self.dataGen.getRandomIntVector(1, 16),
                                  self.dataGen.getRandomFloatVector(1, 0, 1))
        
        facial_forms2 = np.append(self.dataGen.getRandomIntVector(1, 16),
                                  self.dataGen.getRandomFloatVector(1, 0, 1))
        assert len(facial_forms1) == len(facial_forms1) == 2

        featureList.append(facial_forms1)
        featureList.append(facial_forms2)

        # Create feature vector
        features = np.array([])
        for i in featureList:
            features = np.append(features, i)

        return features

    def decodeOutput(self, x):
        """ Function to decode the output vector of the  ML model or example data such as outputs from ```generateDataExample()``` 

        Args:
            x (np.array): Array of encoded in-game settings

        Returns:
            out (np.array): Array of decoded in-game settings
        """

        def _oneHotToInt(vector):
            # Turns OneHot encoded vectors to their in-game preset number
            return int(np.argmax(vector)+1)

        def _oneHotToString(vector):
            # Decode the encoded headshape refine direction vector and intensity
            AEQ = np.array_equal

            # the possible directions
            directions = ["left", "right", "up", "down"]

            # The norm in this encoding is equal to the intensity
            # and using it to calculate the unit vector will give the direction
            norm = np.linalg.norm(vector)
            print("norm", norm)
            if norm == 0:
                return "None", None
            else:
                unitVector = vector / norm
                print("UNIT", unitVector)
                if AEQ(unitVector, np.array([-1, 0])):
                    return directions[0], norm

                elif AEQ(unitVector, np.array([1, 0])):
                    return directions[1], norm

                elif AEQ(unitVector, np.array([0, 1])):
                    return directions[2], norm

                elif AEQ(unitVector, np.array([0, -1])):
                    return directions[3], norm
        
        def _genderIntToString(integer):
            assert integer==0 or integer==1
            if integer==0:
                return "male"
            elif integer==1:
                return "female"

        # Body
        r = x[0]
        angle = np.round(math.atan2(x[2], x[1]) * (180/np.pi), 0)
        # Correct for atan outputting negative angles past 180
        if np.sign(angle) == -1:
            angle = 360 - abs(angle)
        gender = _genderIntToString(x[3])
        skintone = _oneHotToInt(x[4:13])
        body = [r, angle, gender, skintone]

        # Headshape
        headshapePreset = _oneHotToInt(x[13:22])
        headshapeRefineDir, headshapeRefineIntensity = _oneHotToString(x[22:24])
        headshape = [headshapePreset,
                     headshapeRefineDir, headshapeRefineIntensity]

        # Eyes
        eyes = x[24:37]

        # Eye Colour
        eyeColourPreset = _oneHotToInt(x[37:49])
        eyeColour = [eyeColourPreset]

        # Eyebrows
        eyebrowsPreset = _oneHotToInt(x[49:59])
        eyebrowsSliders = x[59:62]
        eyebrows = [eyebrowsPreset] + list(eyebrowsSliders)

        # Forehead
        forehead = x[62:73]

        # Nose
        nose = x[73:91]

        # Ears
        ears = x[91:103]

        # Cheeks
        cheeks = x[103:115]

        # Mouth
        mouth = x[115:128]

        # Jaw
        jaw = x[128:140]

        # Chin
        chin = x[140:152]

        # Neck
        neck = x[152:163]

        # Facial forms
        FF1Preset = _oneHotToInt(x[163:179])
        FF1Intensity = x[179]
        FF2Preset = _oneHotToInt(x[180:196])
        FF2Intensity = x[196]
        FF1 = [FF1Preset, FF1Intensity]
        FF2 = [FF2Preset, FF2Intensity]

        out_vals = [body, headshape, eyes, eyeColour, eyebrows,
                    forehead, nose, ears, cheeks, mouth, jaw, chin, neck, FF1, FF2]
        out_dict = {"body": None, "headshape": None, "eyes": None, "eye_colour": None,
                    "eyebrows": None, "forehead": None, "nose": None, "ears": None,
                    "cheeks": None, "mouth": None, "jaw": None, "chin": None,
                    "neck": None, "facial_forms1": None, "facial_forms2": None}
        for val, key in zip(out_vals, out_dict.keys()):
            out_dict[key] = list(val)
        return out_dict

    def generateEncodedDataExample(self):
        encodedOutput = self.encodeInput(self.generateDataExample())
        return np.round(encodedOutput, 2)

    def generateGTDataset(self,n_samples=10):
        startIdx = 0
        imgPath = os.path.join(self.dataPath,"ingame_images")
        labelAidPath = os.path.join(self.dataPath,"labelling_aid")
        
        if not os.path.exists(labelAidPath):
            os.mkdir(labelAidPath)
        
        # Check if dataJSON already exists and set the starting index to start after
        # the current highest entry
        if os.path.exists(os.path.join(self.dataPath,self.dataJsonFileName)):
            with open(os.path.join(self.dataPath,self.dataJsonFileName),"r") as f:
                dataJSON = json.load(f)
                lastID = int(list(dataJSON.keys())[-1])
                startIdx = lastID + 1 
        else:
            dataJSON = {}
        
        # Check if ground truth already exists 
        if os.path.exists(os.path.join(self.dataPath,"ground_truth",self.gtNPYFileName)):
            dataset = np.load(os.path.join(self.dataPath,"ground_truth",self.gtNPYFileName))
        else:
            dataset= np.empty((0,197))
            
        for i in range(startIdx, startIdx+n_samples):
            # Generate an example
            example = self.generateEncodedDataExample()
            
            # Decode the example
            decodedExample = self.decodeOutput(example)
            
            # Logging Data to a JSON
            dataEntry = {"image":str(os.path.join(imgPath,f"{i}",f"{i}.png")),
                         "deepface_emb_ID":i,
                         "ground_truth_ID":i,
                         "data":decodedExample}
            
            dataJSON[f"{i}"] = dataEntry
            
            #Create labelling aid - Radial and Cross selectors per example
            examplePath = os.path.join(labelAidPath,f"{i}")
            if not os.path.exists(examplePath):
                os.mkdir(examplePath)
                
            #Save the images corressponding to the example
            radial = bodyToImg(r=decodedExample["body"][0],angle=decodedExample["body"][1])
            cross = headshapeToImg(direction=decodedExample["headshape"][1],intensity=decodedExample["headshape"][2])
            cv.imwrite(os.path.join(examplePath, f"body_{i}.png"),radial)
            cv.imwrite(os.path.join(examplePath, f"headshape_{i}.png"),cross)
            
            #Append the example to the dataset
            example = np.expand_dims(example,axis=0)
            dataset = np.append(dataset,example,axis=0)
                
        # Saving the ground_truth data
        np.save(os.path.join(self.dataPath,"ground_truth",self.gtNPYFileName),dataset)
        
        # Save the data.json
        with open(os.path.join(self.dataPath,self.dataJsonFileName), "w") as f:
            json.dump(dataJSON,f)
        
        print(f"Dataset generaeted - dataset JSON in {self.dataPath} and ground truth vectors in {os.path.join(self.dataPath,'ground_truth')}")
        print(f"Labelling aids (radial and cross selectors for BODY and HEADSHAPE respectively) have been saved in {labelAidPath} ")
        return dataset

    def generateDeepFaceData(self):
        from deepface import DeepFace
        # In order for embeddings to be genereated, first we need to make sure that the data was generated
        # DeepFace will throw an error if input images dont exist
        assert os.path.exists(os.path.join(self.dataPath,self.dataJsonFileName)), \
        "The ground truth data needs to be generated first, run generateGTDataset"
        
        # Load the data JSON
        embPath = os.path.join(self.dataPath,"embeddings")
        with open(os.path.join(self.dataPath,self.dataJsonFileName), "r") as f:
            dataJSON = json.load(f)
        
        # Check for existing embeddings and find starting index
        if os.path.exists(os.path.join(embPath,"embeddings.npy")):
            embeddings = np.load(os.path.join(embPath,"embeddings.npy"))
            startIdx = len(embeddings)
        else:
            embeddings = np.empty((0,2622)) # TODO add a lookup table for each model's representation length
            startIdx = 0
        
        # Generate embeddings using deepface
        new_entries = list(dataJSON.keys())[startIdx:]
        if new_entries == []:
            print("\n\nNo new entries detected. Skipping embedding generation\n\n")
            return 0
        else:
            for key in tqdm(list(dataJSON.keys())[startIdx:],f"Generating face embeddings. Start index: {startIdx}"):
                embedding = DeepFace.represent(dataJSON[key]["image"],model_name=self.model)
                embedding = np.expand_dims(embedding[0]["embedding"],axis=0)
                embeddings = np.append(embeddings,embedding,axis=0)
        
            #Save the embeddings
            embeddings = np.array(embeddings)
            np.save(os.path.join(embPath,"embeddings"),embeddings)
            print(f"Embeddings Generated in {embPath}. Training can be run now")
    
            return 0
    
if __name__ == "__main__":
    a = DataGen()
    pp = pprint.PrettyPrinter(indent=4)
    test = False
        
    if test:
        # Generate one example
        example = a.generateDataExample()
        print("UNPROCESSED\n", example)
    
        # Generate Dict
        exampleDict = a.inputData2Dict(example)
        print("EXAMPLE DICT\n", exampleDict)
    
        # Encode the generated example for use in machine learning
        encodedInput = a.encodeInput(example)
        print("MANUALLY ENCODED INPUT\n", encodedInput)
    
        # # Alternatively, can also use
        # encodedInput2 = a.generateEncodedDataExample()
        # print("ALTERNATIVE\n", encodedInput2)
    
        # Decode the encoded input/ output from ML model
        decodedOutput = a.decodeOutput(encodedInput)
        print("DECODED")
        pp.pprint(decodedOutput)

        # Generate a radial selector image from body values and cross-shaped selector from headshape values
        print("BODY: ", decodedOutput["body"])
        bodyToImg(r=decodedOutput["body"][0],angle=decodedOutput["body"][1])
        headshapeToImg(direction=decodedOutput["headshape"][1],intensity=decodedOutput["headshape"][2])
    else:
        print(a.arrayShapes())
        # a.generateGTDataset(10)
        # a.generateDeepFaceData()
        print(a.arrayShapes())
