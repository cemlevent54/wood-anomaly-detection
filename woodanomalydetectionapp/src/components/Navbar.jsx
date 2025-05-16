import React, { useState, useEffect } from "react";
import "../css/Navbar.css";
import { Link } from "react-router-dom";

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);
  const [modelsOpen, setModelsOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768);

  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 768);
      if (window.innerWidth > 768) {
        setMenuOpen(false);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  const toggleMenu = () => setMenuOpen(!menuOpen);

  const models = [
    { name: "Efficient AD", path: "/models/efficient-ad" },
    { name: "Revisiting Reverse Dissilation", path: "/models/revisiting-reverse-dissilation" },
    { name: "Uninet", path: "/models/uninet" },
    { name: "STPM", path: "/models/stpm" }
  ];

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-logo">Wood Anomaly Detection</div>

        {/* Hamburger Icon */}
        <div className="hamburger" onClick={toggleMenu}>
          ☰
        </div>

        <div className={`navbar-links ${menuOpen ? "active" : ""}`}>
          <Link to="/" onClick={() => isMobile && setMenuOpen(false)}>Home</Link>
          <Link to="/about" onClick={() => isMobile && setMenuOpen(false)}>About</Link>
          <Link to="/howto" onClick={() => isMobile && setMenuOpen(false)}>How to Use</Link>
          
          
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

