# Physical-Information-Driven-and-Guided-Multimodal-Multi-task-Neural-Network
Physical Information-Driven and Guided Multimodal Multi-task Neural Network (PIMNN)

# Code Explanation:
'mutilcontrolValueAndStabilityModelUnetPINNRE.py'represents the model code of the PIMNN.

'mutilcontrolValueAndStabilityModelUnetRENonPLoss.py'represents the model code obtained 
after PIMNN removed the physical equations from the loss function.

'mutilcontrolValueAndStabilityModelUnetRENonInPhysic.py'represents the model code obtained 
after PIMNN removed the internal physical equations from the model.

'mutilcontrolValueAndStabilityModelUnetRE.py'represents the model code obtained 
after removing all the physical equations from both the internal physical equations of the PIMNN model and the loss function.

'mutilcontrolVmutilcontrolValueAndStabilityModelUnetPINNRE_ConvGRU' represents 
the code for replacing the stability prediction branch part of 
the model 'mutilcontrolVmutilcontrolValueAndStabilityModelUnetPINNRE' with the ConvGRU model.

'mutilcontrolVmutilcontrolValueAndStabilityModelUnetPINNRE_ConvLSTM' represents 
the code for replacing the stability prediction branch part of 
the model 'mutilcontrolVmutilcontrolValueAndStabilityModelUnetPINNRE' with the ConvLSTM model.

# Dataset Usage Instructions:
The compressed files in the dataset need to be fully decompressed into the directory of the dataset file.

# Code experimental environment:
The code is run under Python 3.11.11 and PyTorch 2.5.1 versions. 
The detailed library file versions can be found in the requirements file. 
The recommended code can be run on NVIDIA graphics cards with video memory of at least 24GB.1
