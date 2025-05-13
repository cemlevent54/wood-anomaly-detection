import React from "react";
import "../css/HowToUse.css";

const HowToUse = () => {
  return (
    <div className="how-to-use-container">
      <h1>Kullanım Kılavuzu</h1>
      
      <div className="instruction-section">
        <h2>1. Görsel Yükleme</h2>
        <p>
          - "Dosya Seç" butonuna tıklayarak analiz etmek istediğiniz ahşap görselini yükleyin.
          - Desteklenen formatlar: JPG, PNG, JPEG
          - Yüklenen görseli büyütmek için "Büyüt" butonunu kullanabilirsiniz.
        </p>
      </div>

      <div className="instruction-section">
        <h2>2. Model Seçimi</h2>
        <p>
          Aşağıdaki modellerden birini seçin:
          - Efficient AD
          - Revisiting Reverse Dissilation
          - Uninet
          - STPM
        </p>
      </div>

      <div className="instruction-section">
        <h2>3. Analiz Başlatma</h2>
        <p>
          - "GET RESULTS" butonuna tıklayarak analizi başlatın.
          - Sistem seçilen modeli kullanarak görseli analiz edecektir.
          - Analiz sonuçları otomatik olarak görüntülenecektir.
        </p>
      </div>

      <div className="instruction-section">
        <h2>4. Sonuçları İnceleme</h2>
        <p>
          - Analiz sonuçları sağ tarafta görüntülenecektir.
          - Sonuç görselini büyütmek için "Büyüt" butonunu kullanabilirsiniz.
          - Sonuçlar şunları içerir:
            * Anomali tespiti
            * F1 Skoru
            * IOU Skoru
            * Genel Skor
        </p>
      </div>

      <div className="instruction-section">
        <h2>5. Sonuçları Dışa Aktarma</h2>
        <p>
          - "Export" butonunu kullanarak analiz sonuçlarını dışa aktarabilirsiniz.
          - Sonuçlar otomatik olarak indirilecektir.
        </p>
      </div>

      <div className="note-section">
        <h3>Önemli Notlar:</h3>
        <ul>
          <li>Analiz için internet bağlantısı gereklidir.</li>
          <li>Büyük boyutlu görseller işlenirken biraz zaman alabilir.</li>
          <li>En iyi sonuçlar için yüksek kaliteli görseller kullanmanız önerilir.</li>
        </ul>
      </div>
    </div>
  );
};

export default HowToUse;
