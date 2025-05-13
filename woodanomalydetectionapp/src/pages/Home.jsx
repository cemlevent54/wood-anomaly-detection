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
  const [showModal, setShowModal] = useState(false); // Yüklenen resim için modal
  const [showResultModal, setShowResultModal] = useState(false); // Sonuç resmi için modal

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
      alert("Lütfen bir model seçiniz ve bir resim yükleyiniz.");
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
      alert("Model testi sırasında bir hata oluştu.");
    }
  };

  const handleExport = () => {
    alert("Sonuçlar dışa aktarıldı!");
  };

  return (
    <div className="container">
      <div className="image-section">
        <div className="image-box">
          {uploadedImage ? (
            <img src={uploadedImage} alt="Uploaded" />
          ) : (
            "YÜKLENEN RESİM"
          )}
        </div>
        <ImageUploader onUpload={handleUpload} />
        {uploadedImage && (
          <button style={{ marginTop: "10px" }} onClick={() => setShowModal(true)}>
            Büyüt
          </button>
        )}
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
        {resultImage && (
          <button style={{ marginTop: "10px" }} onClick={() => setShowResultModal(true)}>
            Büyüt
          </button>
        )}
        {/*<MetricsDisplay metrics={metrics} />*/}
      </div>

      {/* Modal: Yüklenen resim */}
      {showModal && (
        <div style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100vw",
          height: "100vh",
          background: "rgba(0,0,0,0.6)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 1000
        }}
          onClick={() => setShowModal(false)}
        >
          <div style={{
            background: "#fff",
            padding: 20,
            borderRadius: 10,
            boxShadow: "0 4px 32px rgba(0,0,0,0.2)",
            maxWidth: "90vw",
            maxHeight: "90vh",
            display: "flex",
            flexDirection: "column",
            alignItems: "center"
          }}
            onClick={e => e.stopPropagation()}
          >
            <img src={uploadedImage} alt="Büyük Görsel" style={{
              maxWidth: "80vw",
              maxHeight: "80vh",
              borderRadius: 8
            }} />
            <button style={{ marginTop: 16 }} onClick={() => setShowModal(false)}>
              Kapat
            </button>
          </div>
        </div>
      )}
      {/* Modal: Sonuç resmi */}
      {showResultModal && (
        <div style={{
          position: "fixed",
          top: 0,
          left: 0,
          width: "100vw",
          height: "100vh",
          background: "rgba(0,0,0,0.6)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          zIndex: 1000
        }}
          onClick={() => setShowResultModal(false)}
        >
          <div style={{
            background: "#fff",
            padding: 20,
            borderRadius: 10,
            boxShadow: "0 4px 32px rgba(0,0,0,0.2)",
            maxWidth: "90vw",
            maxHeight: "90vh",
            display: "flex",
            flexDirection: "column",
            alignItems: "center"
          }}
            onClick={e => e.stopPropagation()}
          >
            <img src={resultImage} alt="Büyük Sonuç Görseli" style={{
              maxWidth: "80vw",
              maxHeight: "80vh",
              borderRadius: 8
            }} />
            <button style={{ marginTop: 16 }} onClick={() => setShowResultModal(false)}>
              Kapat
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Home;
