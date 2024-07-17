To perform context recognition from short voice conversations, we can convert audio to text and apply natural language processing (NLP) techniques to analyze the textual content. However, the existing text classification systems that apply NLP techniques to extract meaningful information from texts require large training data. In this paper, we propose a novel framework to extract contexts from short-spoken texts requiring smaller training datasets. This framework exploits the power of transfer learning and uses a fully connected neural network aided with SBERT encoding, and an attention mechanism. Our proposed framework has been evaluated using two datasets containing short smart home commands. Evaluation results demonstrate that our model achieves higher accuracy in context recognition with low computational costs and less training time compared to other methods like BERT and deep neural networks.


**Getting Started**

To get started with this project, follow these steps:

Clone the Repository: git clone https://github.com/Jeniya1378/Context-Recognition-from-Voice-Commands-in-Smart-Home

Install Dependencies: Navigate to the project directory and install the necessary dependencies using pip install -r requirements.txt.

Run the System: Execute the main script to start the voice command recognition system.

There are several supporting files and datasets to evaluate the performance. When running the main.py there will be options given to choose which file to run. 
If you want to run the model developed for this project, please select "4" and then "6". This will run the "SAF.py" file. 


**Contribution**

We welcome contributions from the community. If you are interested in contributing, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

**License**

Licensed under the MIT License. See the LICENSE file for details.

**Citation**

If you use this code or dataset in your research, please cite:

@inproceedings{sultana2024framework,
  title={A Framework for Context Recognition from Voice Commands and Conversations with Smart Assistants},
  author={Sultana, Jeniya and Iqbal, Razib},
  booktitle={2024 IEEE 21st Consumer Communications \& Networking Conference (CCNC)},
  pages={218--221},
  year={2024},
  organization={IEEE}
}

