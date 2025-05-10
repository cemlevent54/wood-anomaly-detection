import React, { useState } from "react";
import ImageUploader from "../components/ImageUploader";
import ImageSelector from "../components/ImageSelector";
import ResultDisplay from "../components/ResultDisplay";
import MetricsDisplay from "../components/MetricsDisplay";
import ExportButton from "../components/ExportButton";
import "../css/Home.css";

const Home = () => {
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [selectedModel, setSelectedModel] = useState("");
  const [resultImage, setResultImage] = useState(null); // This will hold the decoded overlay
  const [metrics, setMetrics] = useState(null);

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadedFile(file);
      setUploadedImage(URL.createObjectURL(file));
    }
  };

  const handleModelSelect = (e) => {
    setSelectedModel(e.target.value);
  };

  const handleGetResults = async () => {
    if (!uploadedFile || !selectedModel) {
      alert("Please select a model and upload an image.");
      return;
    }

    const formData = new FormData();
    formData.append("file", uploadedFile);
    formData.append("model_name", selectedModel);

    try {
      const response = await fetch("http://localhost:8020/test/test_model", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Model request failed.");
      }

      const result = await response.json();

      // Only use overlay_base64 to display the result image
      const overlayDataUrl = `${result.overlay_base64}`;
      setResultImage(overlayDataUrl);

      // Set metrics manually from response fields
      setMetrics({
        F1: result.f1_score,
        IOU: result.iou_score,
        score: result.score,
        anomaly: result.prediction === "anomaly",
      });
    } catch (error) {
      console.error("Error:", error);
      alert("An error occurred while testing the model.");
    }
  };

  const handleExport = () => {
    alert("Results exported!");
  };

  return (
    <div className="container">
      <div className="image-section">
        <div className="image-box">
          {uploadedImage ? (
            <img src={uploadedImage} alt="Uploaded" />
          ) : (
            "UPLOADED IMAGE"
          )}
        </div>
        <ImageUploader onUpload={handleUpload} />
      </div>

      <div className="control-section">
        <ImageSelector
          options={[
            { label: "Efficient AD", value: "efficient_ad" },
            { label: "Revisiting Reverse Dissilation", value: "revisiting_reverse_dissilation" },
            { label: "Uninet", value: "uninet" },
            { label: "STPM", value: "stpm" },
          ]}
          onSelect={handleModelSelect}
        />
        <button onClick={handleGetResults}>GET RESULTS</button>
        <ExportButton onClick={handleExport} />
      </div>

      <div className="result-section">
        <ResultDisplay resultImage={resultImage} />
        <MetricsDisplay metrics={metrics} />
      </div>
    </div>
  );
};

export default Home;
