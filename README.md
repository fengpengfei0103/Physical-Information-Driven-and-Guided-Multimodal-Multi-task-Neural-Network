# Physical-Information-Driven-and-Guided-Multimodal-Multi-task-Neural-Network
Physical Information-Driven and Guided Multimodal Multi-task Neural Network (PMNN)

<img width="3425" height="2450" alt="Image" src="https://github.com/user-attachments/assets/d5e5ece8-ae49-4112-b17b-5b7c55ea00c7" />

# Code Explanation:
- 'mutilcontrolValueAndStabilityModelUnetPINNRE.py'represents the model code of the PMNN.
- 'mutilcontrolValueAndStabilityModelUnetRENonPLoss.py'represents the model code obtained after PMNN removed the physical equations from the loss function.
- 'mutilcontrolValueAndStabilityModelUnetRENonInPhysic.py'represents the model code obtained after PMNN removed the internal physical equations from the model.
- 'mutilcontrolValueAndStabilityModelUnetRE.py'represents the model code obtained after removing all the physical equations from both the internal physical equations of the PMNN model and the loss function.
- 'mutilcontrolVmutilcontrolValueAndStabilityModelUnetPINNRE_ConvGRU' represents the code for replacing the stability prediction branch part of the model 'mutilcontrolVmutilcontrolValueAndStabilityModelUnetPINNRE' with the ConvGRU model.
- 'mutilcontrolVmutilcontrolValueAndStabilityModelUnetPINNRE_ConvLSTM' represents the code for replacing the stability prediction branch part of the model 'mutilcontrolVmutilcontrolValueAndStabilityModelUnetPINNRE' with the ConvLSTM model.
- 'mutilcontrolValueAndStabilityModelUnetPINNRE_FineTuning.py' represents a two-stage training strategy is employed: simulation data are first applied pre-train the network, and real monitoring data are subsequently utilized for fine-tuning, thereby enhancing the generalizability and applicability of the framework to real-world conditions.

# Dataset Usage Instructions:
The compressed files in the dataset need to be fully decompressed into the directory of the dataset file.

# Code experimental environment:
- The code is run under Python 3.11.11 and PyTorch 2.5.1 versions.
- The detailed library file versions can be found in the requirements file.
- The recommended code can be run on NVIDIA graphics cards with video memory of at least 24GB.
