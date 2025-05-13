import React, { useState } from "react";
import "../css/Navbar.css";
import { Link } from "react-router-dom";

const Navbar = () => {
  const [menuOpen, setMenuOpen] = useState(false);

  const toggleMenu = () => setMenuOpen(!menuOpen);

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-logo">Wood Anomaly Detection</div>

        {/* Hamburger Icon */}
        <div className="hamburger" onClick={toggleMenu}>
          ☰
        </div>

        <div className={`navbar-links ${menuOpen ? "active" : ""}`}>
          <Link to="/">Anasayfa</Link>
          <Link to="/about">Hakkında</Link>
          <Link to="/howto">Nasıl kullanılır?</Link>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;

