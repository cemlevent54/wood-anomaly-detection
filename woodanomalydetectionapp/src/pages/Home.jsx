import React, { useState } from "react";
import ImageUploader from "../components/ImageUploader";
import ImageSelector from "../components/ImageSelector";
import ResultDisplay from "../components/ResultDisplay";
import MetricsDisplay from "../components/MetricsDisplay";
import ExportButton from "../components/ExportButton";
import "../css/Home.css";  // Eğer özel stil gerekiyorsa

const Home = () => {
  const [uploadedImage, setUploadedImage] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [metrics, setMetrics] = useState(null);

  const handleUpload = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setUploadedImage(URL.createObjectURL(e.target.files[0]));
    }
  };

  const handleGetResults = () => {
    setResultImage("/sample_result.png");
    setMetrics({ F1: 0.87, IOU: 0.75, AUC: 0.91, anomaly: true });
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
          options={["Model 1", "Model 2"]}
          onSelect={(e) => console.log("Selected:", e.target.value)}
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
