import React from "react";
import "../css/About.css";

const About = () => {
  return (
    <div className="about-container">
      <div className="about-header">
        <h1>Wood Anomaly Detection</h1>
      </div>

      <div className="about-content">
        <section className="about-section">
          <h2>About the Project</h2>
          <p>
            Wood Anomaly Detection is an deep learning-based system developed to detect
            abnormal conditions on wood surfaces. This project aims to optimize production processes
            by automating quality control of wood products.
          </p>
        </section>

        <section className="about-section">
          <h2>Technologies Used</h2>
          <div className="tech-grid">
            <div className="tech-item">
              <h3>Anomaly Detection Models</h3>
              <ul>
                <li>Efficient AD</li>
                <li>Revisiting Reverse Distillation</li>
                <li>Uninet</li>
                <li>STPM</li>
              </ul>
            </div>
            <div className="tech-item">
              <h3>Frontend</h3>
              <ul>
                <li>React.js</li>
                <li>CSS3</li>
                <li>Responsive Design</li>
              </ul>
            </div>
            <div className="tech-item">
              <h3>Backend</h3>
              <ul>
                <li>Python</li>
                <li>FastAPI</li>
                <li>TensorFlow/PyTorch</li>
              </ul>
            </div>
          </div>
        </section>

        <section className="about-section">
          <h2>Code Repository</h2>
          <p>
            You can find the code repository of the project on <a href="https://github.com/cemlevent54/wood-anomaly-detection" target="_blank" rel="noopener noreferrer">GitHub</a>.
          </p>
        </section>

        <section className="about-section">
          <h2>Contact</h2>
          <p>
            You can contact us for more information about the project or to collaborate.
          </p>
          <div className="contact-info">
            <p>Email: <a href="mailto:cemlevent54@gmail.com">cemlevent54@gmail.com</a></p>
            <p>Github: <a href="https://github.com/cemlevent54" target="_blank" rel="noopener noreferrer">https://github.com/cemlevent54</a></p>
            <p>Linkedin: <a href="https://www.linkedin.com/in/leventavci54" target="_blank" rel="noopener noreferrer">https://www.linkedin.com/in/leventavci54</a></p>
            <p>Web: <a href="https://cemleventportfolio.vercel.app/" target="_blank" rel="noopener noreferrer">https://cemleventportfolio.vercel.app/</a></p>
          </div>
        </section>
      </div>
    </div>
  );
};

export default About;
