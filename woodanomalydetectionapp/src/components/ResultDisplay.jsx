import React from 'react';

const ResultDisplay = ({ resultImage }) => {
  return (
    <div className="image-box">
      {resultImage ? <img src={resultImage} alt="Result" /> : "RESULT IMAGE"}
    </div>
  );
};

export default ResultDisplay;
