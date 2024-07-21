<h1>Automated Detection of Glaucoma Using Artificial Intelligence</h1>




<h2>Abstract</h2>
<par> Glaucoma is an ophthalmologic condition that damages the optic nerve head, specifically the optic cup and the optic disc, which can lead to blindness. It is the second most common cause of vision loss. Examining the optic nerve head regions, including the neuroretinal rim area (obtained by subtracting the cup from the disc), is critical for diagnosis. </par>




<h2>Introduction</h2>
<par> This project presents an automated glaucoma detection solution using fundus images and artificial intelligence. The solution uses a two-step approach: AI initially identifies the contours of the optic cup and disc, then uses this data to derive an automated decision rule for glaucoma detection. </par>




<h2>First Step: Segmentation</h2>
<par> The initial step uses a UNET architecture with three output channels to process fundus eye images, segmenting the optic cup (black) and optic disc (gray) to generate a mask. </par>

![unet](https://github.com/user-attachments/assets/cdb9cc0c-884d-47c7-aff6-099f75061a95)




<h2>Second Step: Classification</h2>
<par> This step unfolds into three distinct phases: </par>



<h3>Phase One: Extract Metrics</h3>
<par> Metrics such as the cup to disc ratio (CDR), horizontal/vertical cup to disc diameter (hCDR/vCDR), and neuroretinal rim area ratio (NRR) based on the ISNT (inferior, superior, nasal, temporal) areas are computed from the ROI and recorded in a CSV file. </par>

![roi-metrics](https://github.com/user-attachments/assets/2733f7db-fb51-4b88-b71a-1ecdb84ab440)



<h3>Phase Two: Choose Classification Models</h3>
<par> Several classification models are used: linear-based, decision tree-based, and kernel-based. Each model generates a prediction, and the final diagnosis is determined using majority voting. </par>



<h3>Phase Three: Explainability</h3>
Shapley values are used to generate explainability plots for each of the classification models included. These plots help in understanding the contribution of each feature to the final prediction.

![Gradient_Boosting](https://github.com/user-attachments/assets/39818912-2c72-4454-8354-e8359713bdbf)




<h2>Dataset</h2>
<par> The REFUGE2 dataset, a well-known dataset for segmentation, is used. It can be downloaded from Kaggle.</par>
<href>https://www.kaggle.com/datasets/victorlemosml/refuge2?rvi=1</href>




<h2>Validation</h2>
<par>The segmentation model was trained for 50 epochs, achieving an accuracy of 99.62% based on correctly identified pixels. For the classification models, cross-validation was used, with accuracy ranging from 89.58% to 92.92%.</par>




<h2>Tests</h2>
<par>After execution, a PDF report is generated based on a selected image from the local computer. This report includes the input image, segmented mask, extracted ROI, extracted metrics, and the predicted diagnosis. Additional pages contain explainability plots of the classification metrics used.</par>

![pdf](https://github.com/user-attachments/assets/c5accedc-5b4e-465e-8409-c4bb005038d7)





<h2>Setup</h2>
1. Clone the repository

2. Create a virtual environment (Python 3.10 recommended)

3. Install the required packages

The checkpoint files for the classification and segmentation models, as well as the preprocessed dataset, can be found on my public drive here
<href>https://drive.google.com/drive/folders/1yVPe3TwKF1-Krmc2WfJJNfsAUKlnLWej?usp=sharing</href>
