# Multimodal-Live-Expression-Classifier

Data Sources and Processing
The dataset was sourced from Aff-Wild2 repository. I chose to store all the data using Google Cloud Storage (GCS), chosen over Google Drive due to storage limits. GCS enabled efficient access to 650k+ image annotations.

Processing Steps:

Used pickle for caching data pairs, allowing fast retrieval of large multimodal samples (Expr, VA, AU).
Filtered incomplete annotations (-1 values), reducing to 178,997 valid examples (e.g., Neutral: 59,416; Disgust: 359; Fear: 162).
Addressed imbalance by downsampling majorities (Neutral, Happy, Other), resulting in a 100k balanced dataset.
Split into five 20k subsets for ensemble training; applied class balancers (upscaling to ~56k per subset) and boost factors (e.g., +0.5 for Fear/Surprise).
Used the 4th subset for a balanced 10k validation set and the 5th as a 20k example test set ensuring no data leakage.



From project logs:

Cache1 validation: Expr F1 per class (e.g., Sad: 0.7002, Surprise: 0.2267).
Cache2: Improved balance (e.g., Happy: 0.5880, Disgust: 0.5312).
Cache3: Varied performance (e.g., Sad: 0.7327, Surprise: 0.3522).

Architecture and Development Insights
Early experiments focused on an AU-specific model with feature branching (splitting face into eyes, mouth, nose for specialized learning), but it underperformed. Testing on random samples showed poor AU score alignment with ground-truth faces, leading to a pivot.
Revelation and Pivot: Researched multi-modal modeling and ensembles. With three annotation types, adopted MTL to train on all simultaneously, boosting patterns (e.g., AUs like AU1/AU2 with Surprise; VA like (-0.64, 0.60) for Fear). Initially planned 5-model ensemble (bagging on subsets), but trained 3 models from scratch after overfitting issues with prior checkpoints (e.g., 64 epochs favoring "Other").
Model Architecture:

Base: Transfer learning with EfficientNetV2S (ImageNet weights, unfroze 250 layers).
Fusion: Dropout + Multi-Head Attention for multimodal integration + Batch Normalization + 70% Dropout.
Heads: Shared Dense layer + task-specific branches.
Outputs: Expr (Softmax with kernel regularizers), VA (Tanh), AU (Sigmoid).

Metrics and Monitoring:

Expr/AU: Precision, Recall, F1 (per-class verbose outputs).
VA: Concordance Correlation Coefficient (CCC), equivalent to F1 for continuous values.
Trained on L4 GPUs (cost-efficient; TPUs incompatible on Colab).
Checkpoints saved per epoch for easy resuming.
Learnings: Over-balanced minorities (e.g., reduced Sad boost from 2→1.5); incorporated external datasets for underrepresented classes (Anger, Disgust, Fear, Sad).

From logs: Initial real-time tests (Epoch 49) favored "Other"/"Neutral"; post-training, better detection of "Happy"/"Surprise" at 4-5 FPS.
Results
Evaluated on 20k test set:

Expr Metrics (Macro Avg): Precision: 0.775, Recall: 0.798, F1: 0.781.

Per-Class F1: Neutral: 0.882, Anger: 0.739, Disgust: 0.595, Fear: 0.607, Happy: 0.857, Sad: 0.936, Surprise: 0.718, Other: 0.916.


AU F1: 0.795.
VA CCC: 0.703.
Overall Accuracy: 0.872.

Validation peaks (e.g., Cache2: Expr F1 ~0.534, VA CCC 0.69, AU F1 0.59) showed strong generalization.

Installation

Clone the repository:
textgit clone https://github.com/yourusername/multimodal-emotion-classifier.git
cd multimodal-emotion-classifier

Install dependencies (Python 3.8+ recommended):
textpip install opencv-python numpy tensorflow ultralytics huggingface-hub scipy collections

or use the requirements.txt file

Note: TensorFlow may require GPU setup (e.g., CUDA for NVIDIA). See TensorFlow docs for details.


Download the model:

Create a final_model directory.
Download the .keras file (e.g., full_mtl_model.keras) from [Google Drive link] and place it in final_model/.
The YOLOv11 face model downloads automatically via Hugging Face Hub.



Usage: How to Run the Code

Ensure your webcam is connected (or use a video file by modifying cv2.VideoCapture(0) to cv2.VideoCapture('path/to/video.mp4')).
Run the script:
textpython emotion_classifier.py  # Assuming the code is saved as this

The app will open a window showing live video with bounding boxes and emotion labels.

Press 'q' to quit.
FPS is printed every 10 frames (typically 2-4 FPS on standard hardware).


Customization:

Adjust self-ensemble window in code (e.g., deque(maxlen=3) to 5 for more smoothing).
For multi-model ensembles, load additional .keras files and average predictions.



Contributing
Pull requests welcome! For major changes, open an issue first. Ensure code follows PEP8 and includes tests.
License
MIT License. See LICENSE for details.
Acknowledgments

Inspired by research on MTL for emotion recognition (e.g., FACS for AUs, VA circumplex model).
Tools: TensorFlow, Ultralytics YOLO, Hugging Face.

References:
• D. Kollias, et. al.: "Advancements in Affective and Behavior Analysis: The 8th ABAW Workshop and
Competition", 2025
@article{Kollias2025, author = "Dimitrios Kollias and Panagiotis Tzirakis and Alan S. Cowen and Stefanos Zafeiriou and Irene
Kotsia and Eric Granger and Marco Pedersoli and Simon L. Bacon and Alice Baird and Chris Gagne and Chunchang Shao and
Guanyu Hu and Soufiane Belharbi and Muhammad Haseeb Aslam", title = "{Advancements in Affective and Behavior
Analysis: The 8th ABAW Workshop and Competition}", year = "2025", doi = "10.6084/m9.figshare.28524563.v4"}
@article{kolliasadvancements, title={Advancements in Affective and Behavior Analysis: The 8th ABAW Workshop and
Competition}, author={Kollias, Dimitrios and Tzirakis, Panagiotis and Cowen, Alan and Kotsia, Irene and Cogitat, UK and
Granger, Eric and Pedersoli, Marco and Bacon, Simon and Baird, Alice and Shao, Chunchang and others}}
• D. Kollias, et. al.: "7th abaw competition: Multi-task learning and compound expression recognition", ECCV
2024
@article{kollias20247th,title={7th abaw competition: Multi-task learning and compound expression
recognition},author={Kollias, Dimitrios and Zafeiriou, Stefanos and Kotsia, Irene and Dhall, Abhinav and Ghosh, Shreya and
Shao, Chunchang and Hu, Guanyu},journal={arXiv preprint arXiv:2407.03835},year={2024}}
• D. Kollias, et. al.: "The 6th Affective Behavior Analysis in-the-wild (ABAW) Competition", IEEE CVPR 2024
@inproceedings{kollias20246th,title={The 6th affective behavior analysis in-the-wild (abaw) competition},author={Kollias,
Dimitrios and Tzirakis, Panagiotis and Cowen, Alan and Zafeiriou, Stefanos and Kotsia, Irene and Baird, Alice and Gagne,
Chris and Shao, Chunchang and Hu, Guanyu},booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition},pages={4587--4598},year={2024}}

• D. Kollias, et. al.: "Distribution matching for multi-task learning of classification tasks: a large-scale study on
faces & beyond ", AAAI 2024

@inproceedings{kollias2024distribution,title={Distribution matching for multi-task learning of classification tasks: a large-
scale study on faces \& beyond},author={Kollias, Dimitrios and Sharmanska, Viktoriia and Zafeiriou,

Stefanos},booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},volume={38},number={3},pages={2813-
-2821},year={2024}}
• D. Kollias, et. al.: "ABAW: Valence-Arousal Estimation, Expression Recognition, Action Unit Detection &
Emotional Reaction Intensity Estimation Challenges", IEEE CVPR 2023
@inproceedings{kollias2023abaw2, title={Abaw: Valence-arousal estimation, expression recognition, action unit detection
\& emotional reaction intensity estimation challenges}, author={Kollias, Dimitrios and Tzirakis, Panagiotis and Baird, Alice
and Cowen, Alan and Zafeiriou, Stefanos}, booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and
Pattern Recognition}, pages={5888--5897}, year={2023} }
• D. Kollias: "Multi-Label Compound Expression Recognition: C-EXPR Database & Network", IEEE CVPR 2023
@inproceedings{kollias2023multi, title={Multi-Label Compound Expression Recognition: C-EXPR Database \& Network},
author={Kollias, Dimitrios}, booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern
Recognition}, pages={5589--5598}, year={2023}}
• D. Kollias: " ABAW: Learning from Synthetic Data & Multi-Task Learning Challenges", ECCV 2022
@inproceedings{kollias2023abaw, title={Abaw: Learning from synthetic data \& multi-task learning challenges},
author={Kollias, Dimitrios}, booktitle={European Conference on Computer Vision}, pages={157--172}, year={2023},
organization={Springer}}

• D. Kollias: " ABAW: Valence-Arousal Estimation, Expression Recognition, Action Unit Detection & Multi-
TaskLearning Challenges ". IEEE CVPR, 2022

@inproceedings{kollias2022abaw, title={Abaw: Valence-arousal estimation, expression recognition, action unit detection
\& multi-task learning challenges}, author={Kollias, Dimitrios}, booktitle={Proceedings of the IEEE/CVF Conference
onComputer Vision and Pattern Recognition}, pages={2328--2336}, year={2022} }
• D. Kollias, S. Zafeiriou: "Analysing Affective Behavior in the second ABAW2 Competition". ICCV, 2021
@inproceedings{kollias2021analysing, title={Analysing affective behavior in the second abaw2
competition}, author={Kollias, Dimitrios and Zafeiriou, Stefanos}, booktitle={Proceedings of the IEEE/CVF International
Conference on Computer Vision}, pages={3652--3660}, year={2021}}
• D. Kollias, et. al.: "Analysing Affective Behavior in the First ABAW 2020 Competition". IEEE FG, 2020
@inproceedings{kollias2020analysing, title={Analysing Affective Behavior in the First ABAW 2020 Competition},
author={Kollias, D and Schulc, A and Hajiyev, E and Zafeiriou, S}, booktitle={2020 15th IEEE International Conference on
Automatic Face and Gesture Recognition (FG 2020)(FG)}, pages={794--800}}
• D. Kollias, et. al.: "Distribution Matching for Heterogeneous Multi-Task Learning: a Large-scale Face
Study",2021
@article{kollias2021distribution, title={Distribution Matching for Heterogeneous Multi-Task Learning: a Large-scale Face
Study}, author={Kollias, Dimitrios and Sharmanska, Viktoriia and Zafeiriou, Stefanos}, journal={arXiv preprint
arXiv:2105.03790}, year={2021} }
• D. Kollias, S. Zafeiriou: "Affect Analysis in-the-wild: Valence-Arousal, Expressions, Action Units and a
UnifiedFramework, 2021
@article{kollias2021affect, title={Affect Analysis in-the-wild: Valence-Arousal, Expressions, Action Units and a Unified
Framework}, author={Kollias, Dimitrios and Zafeiriou, Stefanos}, journal={arXiv preprint arXiv:2103.15792}, year={2021}}
• D. Kollias, S. Zafeiriou: "Expression, Affect, Action Unit Recognition: Aff-Wild2, Multi-Task Learning
andArcFace". BMVC, 2019
@article{kollias2019expression, title={Expression, Affect, Action Unit Recognition: Aff-Wild2, Multi-Task Learning and
ArcFace}, author={Kollias, Dimitrios and Zafeiriou, Stefanos}, journal={arXiv preprint arXiv:1910.04855}, year={2019} }
• D. Kollias, et at.: "Face Behavior a la carte: Expressions, Affect and Action Unitsin a Single Network", 2019
@article{kollias2019face,title={Face Behavior a la carte: Expressions, Affect and Action Units in a Single
Network}, author={Kollias, Dimitrios and Sharmanska, Viktoriia and Zafeiriou, Stefanos}, journal={arXiv preprint
arXiv:1910.11111}, year={2019}}

• D. Kollias, et. al.: "Deep Affect Prediction in-the-wild: Aff-Wild Database and Challenge, Deep
Architectures,and Beyond". International Journal of Computer Vision (IJCV), 2019
@article{kollias2019deep, title={Deep affect prediction in-the-wild: Aff-wild database and challenge, deep architectures,
andbeyond}, author={Kollias, Dimitrios and Tzirakis, Panagiotis and Nicolaou, Mihalis A and Papaioannou, Athanasios and
Zhao,Guoying and Schuller, Bj{\"o}rn and Kotsia, Irene and Zafeiriou, Stefanos}, journal={International Journal of Computer
Vision}, pages={1--23}, year={2019}, publisher={Springer} }
• S. Zafeiriou, et. al. "Aff-Wild: Valence and Arousal in-the-wild Challenge". CVPR, 2017
@inproceedings{zafeiriou2017aff, title={Aff-wild: Valence and arousal ‘in-the-wild’challenge}, author={Zafeiriou,
Stefanos and Kollias, Dimitrios and Nicolaou, Mihalis A and Papaioannou, Athanasios and Zhao, Guoying and Kotsia,
Irene}, booktitle={Computer Vision and Pattern Recognition Workshops (CVPRW), 2017 IEEE Conference on},
pages={1980--1987}, year={2017}, organization={IEEE} }
