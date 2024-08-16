import streamlit as st
from PIL import Image
import ultralytics

# Load your saved models here
model_paths = {'Model 1': 'path_to_model_1.pth', 'Model 2': 'path_to_model_2.pth'}  # Add more models as needed   

# Function to load the selected model
def load_model(model_path):
    # Load the model using torch.load or your preferred method
    model= ultralytics.YOLO(model_path)
    return model
# Function to make predictions
def predict(image, model,col2):
    results = model.predict(image)
    #print("Hello")
    col2.image(results[0].plot()[:, :, ::-1], caption="Detected Objects", use_column_width=True)

    

# Main Streamlit app code
def main():
    st.title("Image Prediction App")

    # File upload section
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    # Model selection dropdown
    #model_option = st.selectbox("Select Model", ["Model 1", "Model 2", "Model 3"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        col1,col2=st.columns(2)
        col1.image(image, caption="Uploaded Image", use_column_width=True)

        # Load the selected model
        model = load_model(f"/Users/nishagarg7/Downloads/infotech/automl-git/AutoML_Services/prediction_1/autocv/experiment_7/weights/prediction_1.pt")  # Adjust the model loading path as needed

        # Perform prediction when the user clicks the button
        if st.button("Predict"):
            predict(image, model,col2)
            

# Run the app
if __name__ == "__main__":
    main()
