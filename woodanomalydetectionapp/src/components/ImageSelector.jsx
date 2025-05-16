import React from 'react';

const ImageSelector = ({ options, onSelect }) => {
  return (
    <select onChange={onSelect}>
      <option value="">Select Model</option>
      {options.map((opt, i) => (
        <option key={i} value={opt.value}>{opt.label}</option>
      ))}
    </select>
  );
};

export default ImageSelector;
