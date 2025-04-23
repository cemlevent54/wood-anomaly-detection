import React from 'react';

const ImageUploader = ({ onUpload }) => {
  return (
    <div>
      <input 
        type="file" 
        id="fileUpload" 
        style={{ display: 'none' }} 
        onChange={onUpload} 
      />
      <label htmlFor="fileUpload" className="upload-button">
        Select Image
      </label>
    </div>
  );
};

export default ImageUploader;
