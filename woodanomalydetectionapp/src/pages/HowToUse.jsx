import React from "react";
import "../css/HowToUse.css";

const HowToUse = () => {
  return (
    <div className="how-to-use-container">
      <br /> <br />
      <h1>User Guide</h1>
      
      <div className="instruction-section">
        <h2>1. Image Upload</h2>
        <p>
          - Click the "Select File" button to upload the wood image you want to analyze.
          <br />
          - Supported formats: JPG, PNG, JPEG
          <br />
          - You can use the "Enlarge" button to view the uploaded image in full size.
        </p>
      </div>

      <div className="instruction-section">
        <h2>2. Model Selection</h2>
        <p>
          Choose one of the following models:
          <br />
          - Efficient AD
          <br />
          - Revisiting Reverse Dissilation
          <br />
          - Uninet
          <br />
          - STPM
        </p>
      </div>

      <div className="instruction-section">
        <h2>3. Starting the Analysis</h2>
        <p>
          - Click the "GET RESULTS" button to start the analysis.
          <br />
          - The system will analyze the image using the selected model.
          <br />
          - Analysis results will be displayed automatically.
        </p>
      </div>

      <div className="instruction-section">
        <h2>4. Reviewing Results</h2>
        <p>
          - Analysis results will be displayed on the right side.
          <br />
          - You can use the "Enlarge" button to view the result image in full size.
          <br />
        </p>
      </div>

      

      <div className="note-section">
        <h3>Important Notes:</h3>
        <ul>
          <li>Internet connection is required for analysis.</li>
          <li>Processing large-sized images may take some time.</li>
          <li>It is recommended to use high-quality images for best results.</li>
        </ul>
      </div>
    </div>
  );
};

export default HowToUse;
