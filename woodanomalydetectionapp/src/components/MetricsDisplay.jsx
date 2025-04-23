import React from 'react';

const MetricsDisplay = ({ metrics }) => {
  return (
    <div className="metrics-box">
      {metrics ? (
        <>
          <p>F1 Score: {metrics.F1}</p>
          <p>IOU: {metrics.IOU}</p>
          <p>AUC: {metrics.AUC}</p>
          <p>{metrics.anomaly ? "Anomaly Detected!" : "No Anomaly"}</p>
        </>
      ) : (
        "IMAGE RESULTS"
      )}
    </div>
  );
};

export default MetricsDisplay;
