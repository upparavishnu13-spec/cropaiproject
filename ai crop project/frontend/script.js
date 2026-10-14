/**
 * PlantCare AI - Frontend Interactive Logic
 * Handles single-page navigation, drag-and-drop image uploads,
 * demo samples generator, API inference calls, dashboard telemetry,
 * disease library filtering, and modal dialogs.
 */

// Base API configuration (points to relative /api when served by Flask, or configurable)
const API_BASE = window.location.origin.includes("5500") || window.location.origin.includes("3000")
  ? "http://127.0.0.1:5000"
  : "";

// State
let selectedImageFile = null;
let diseaseLibraryCache = [];
let activeCropFilter = "all";

// DOM Elements
const navLinks = document.querySelectorAll(".nav-item");
const pages = document.querySelectorAll(".page-section");
const dropzone = document.getElementById("dropzone");
const leafInput = document.getElementById("leafInput");
const dropzoneEmpty = document.getElementById("dropzoneEmpty");
const dropzonePreview = document.getElementById("dropzonePreview");
const previewImg = document.getElementById("previewImg");
const previewMetaPill = document.getElementById("previewMetaPill");
const removeImgBtn = document.getElementById("removeImgBtn");
const analyzeBtn = document.getElementById("analyzeBtn");

const resultsEmpty = document.getElementById("resultsEmpty");
const resultsLoading = document.getElementById("resultsLoading");
const resultsPopulated = document.getElementById("resultsPopulated");
const predictionProgressBar = document.getElementById("predictionProgressBar");
const loadingText = document.getElementById("loadingText");

const systemStatusPill = document.getElementById("systemStatusPill");
const globalAlert = document.getElementById("globalAlert");
const globalAlertMsg = document.getElementById("globalAlertMsg");
const globalAlertIcon = document.getElementById("globalAlertIcon");

// ==========================================
// 1. PAGE NAVIGATION & ROUTING
// ==========================================
function navigateTo(pageId) {
  // Update nav tabs
  navLinks.forEach(link => {
    link.classList.toggle("active", link.getAttribute("data-page") === pageId);
  });

  // Switch sections
  pages.forEach(page => {
    const isTarget = page.id === `page-${pageId}`;
    page.classList.toggle("active", isTarget);
  });

  // Close mobile nav drawer if open
  const navContainer = document.getElementById("navLinks");
  if (navContainer) navContainer.classList.remove("mobile-open");

  // Hook specific page load events
  if (pageId === "dashboard") {
    fetchDashboardData();
  } else if (pageId === "library" && diseaseLibraryCache.length === 0) {
    fetchDiseaseLibrary();
  }

  window.scrollTo({ top: 0, behavior: "smooth" });
}

// Mobile menu toggle
const mobileMenuBtn = document.getElementById("mobileMenuBtn");
if (mobileMenuBtn) {
  mobileMenuBtn.addEventListener("click", () => {
    const navContainer = document.getElementById("navLinks");
    if (navContainer) navContainer.classList.toggle("mobile-open");
  });
}

// ==========================================
// 2. SYSTEM HEALTH CHECK
// ==========================================
async function checkSystemHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const statusText = systemStatusPill.querySelector(".status-text");
    if (data.model && data.model.model_loaded) {
      systemStatusPill.className = "system-status-pill online";
      statusText.textContent = `Model Active (${data.model.classes_count} classes)`;
    } else {
      systemStatusPill.className = "system-status-pill demo";
      statusText.textContent = "Simulation / Demo Mode";
      showGlobalAlert(
        "Model weights not yet trained on disk. Demo Mode active. Run 'python train_model.py' to train with your dataset.",
        "ℹ️"
      );
    }
  } catch (err) {
    systemStatusPill.className = "system-status-pill";
    const statusText = systemStatusPill.querySelector(".status-text");
    statusText.textContent = "Backend Offline";
    showGlobalAlert(
      "Cannot connect to Flask backend at http://127.0.0.1:5000. Start backend using: python backend/app.py",
      "⚠️"
    );
  }
}

function showGlobalAlert(msg, icon = "ℹ️") {
  globalAlertMsg.textContent = msg;
  globalAlertIcon.textContent = icon;
  globalAlert.style.display = "block";
}

function closeGlobalAlert() {
  globalAlert.style.display = "none";
}

// ==========================================
// 3. DRAG & DROP / FILE SELECTION
// ==========================================
if (dropzone && leafInput) {
  dropzone.addEventListener("click", (e) => {
    // Only open dialog if clicking empty area, browse link, or icon
    if (!e.target.closest("#removeImgBtn") && dropzonePreview.style.display === "none") {
      leafInput.click();
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleSelectedFile(e.dataTransfer.files[0]);
    }
  });

  leafInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleSelectedFile(e.target.files[0]);
    }
  });

  removeImgBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetDetection();
  });
}

function handleSelectedFile(file) {
  // Validate extension
  const validTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
  const extension = file.name.split(".").pop().toLowerCase();
  const validExtensions = ["jpg", "jpeg", "png", "webp", "bmp"];

  if (!validTypes.includes(file.type) && !validExtensions.includes(extension)) {
    alert("Please select a supported image file (.jpg, .jpeg, .png, .webp, .bmp).");
    return;
  }

  // Validate size (16MB)
  if (file.size > 16 * 1024 * 1024) {
    alert("Image file size exceeds 16 MB. Please upload a smaller photo.");
    return;
  }

  selectedImageFile = file;

  // Render client preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    dropzoneEmpty.style.display = "none";
    dropzonePreview.style.display = "flex";

    // Obtain dimensions
    const tmp = new Image();
    tmp.onload = () => {
      const sizeKb = (file.size / 1024).toFixed(1);
      previewMetaPill.textContent = `${tmp.width} × ${tmp.height} px • ${sizeKb} KB`;
    };
    tmp.src = e.target.result;

    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);
}

// ==========================================
// 4. PRE-PACKAGED SAMPLE GENERATOR
// ==========================================
// Generates realistic colored synthetic leaf samples for immediate testing
function setupSampleChips() {
  const chips = document.querySelectorAll(".sample-chip");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      const sampleType = chip.getAttribute("data-sample");
      generateSampleImage(sampleType);
    });
  });
}

function generateSampleImage(type) {
  const canvas = document.createElement("canvas");
  canvas.width = 300;
  canvas.height = 300;
  const ctx = canvas.getContext("2d");

  // Background
  ctx.fillStyle = "#f5f5f0";
  ctx.fillRect(0, 0, 300, 300);

  // Draw Leaf Base Shape
  ctx.save();
  ctx.translate(150, 150);

  let leafColor = "#388e3c";
  let spotColor = null;
  let sampleName = "leaf_sample.jpg";

  if (type === "tomato_early_blight") {
    leafColor = "#4caf50";
    spotColor = "#5d4037"; // Concentric target spots
    sampleName = "tomato_early_blight_sample.jpg";
  } else if (type === "apple_scab") {
    leafColor = "#2e7d32";
    spotColor = "#263238"; // Dark olive scab
    sampleName = "apple_scab_sample.jpg";
  } else if (type === "potato_blight") {
    leafColor = "#558b2f";
    spotColor = "#3e2723"; // Water soaked dark rot
    sampleName = "potato_late_blight_sample.jpg";
  } else if (type === "tomato_healthy") {
    leafColor = "#2e7d32";
    spotColor = null; // Clean
    sampleName = "tomato_healthy_sample.jpg";
  }

  // Draw Leaf
  ctx.beginPath();
  ctx.moveTo(0, -110);
  ctx.bezierCurveTo(70, -70, 90, 40, 0, 110);
  ctx.bezierCurveTo(-90, 40, -70, -70, 0, -110);
  ctx.fillStyle = leafColor;
  ctx.fill();

  // Draw Central Vein
  ctx.beginPath();
  ctx.moveTo(0, -110);
  ctx.lineTo(0, 110);
  ctx.strokeStyle = "#1b5e20";
  ctx.lineWidth = 3;
  ctx.stroke();

  // Draw Lateral Veins
  for (let y = -70; y < 90; y += 30) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.quadraticCurveTo(35, y - 10, 50, y + 10);
    ctx.moveTo(0, y);
    ctx.quadraticCurveTo(-35, y - 10, -50, y + 10);
    ctx.strokeStyle = "#1b5e20";
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  // Draw Pathogen Spots if applicable
  if (spotColor) {
    const spots = [
      { x: 25, y: -20, r: 16 },
      { x: -30, y: 35, r: 12 },
      { x: 35, y: 55, r: 14 },
      { x: -20, y: -50, r: 10 }
    ];

    spots.forEach(s => {
      // Outer ring
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = spotColor;
      ctx.fill();

      // Inner halo / concentric ring
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r * 0.5, 0, Math.PI * 2);
      ctx.fillStyle = "#8d6e63";
      ctx.fill();
    });
  }

  ctx.restore();

  // Convert canvas to Blob/File
  canvas.toBlob((blob) => {
    const file = new File([blob], sampleName, { type: "image/jpeg" });
    handleSelectedFile(file);
  }, "image/jpeg", 0.95);
}

// ==========================================
// 5. INFERENCE & PREDICTION REQUEST
// ==========================================
if (analyzeBtn) {
  analyzeBtn.addEventListener("click", () => {
    if (!selectedImageFile) return;
    executeLeafPrediction(selectedImageFile);
  });
}

async function executeLeafPrediction(file) {
  // Switch to loading UI
  resultsEmpty.style.display = "none";
  resultsPopulated.style.display = "none";
  resultsLoading.style.display = "flex";
  analyzeBtn.disabled = true;

  // Animate progress simulation for smooth UX
  let progress = 15;
  predictionProgressBar.style.width = "15%";
  loadingText.textContent = "Verifying image structure and resizing to 224x224...";

  const progressInterval = setInterval(() => {
    if (progress < 85) {
      progress += 15;
      predictionProgressBar.style.width = `${progress}%`;
      if (progress === 45) {
        loadingText.textContent = "Running MobileNetV2 deep neural classification...";
      } else if (progress === 75) {
        loadingText.textContent = "Cross-referencing agronomic knowledge base...";
      }
    }
  }, 250);

  try {
    const formData = new FormData();
    formData.append("image", file);

    const response = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      body: formData
    });

    clearInterval(progressInterval);
    predictionProgressBar.style.width = "100%";

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.error || `Server responded with ${response.status}`);
    }

    const data = await response.json();
    setTimeout(() => {
      displayPredictionResults(data);
    }, 300);

  } catch (error) {
    clearInterval(progressInterval);
    resultsLoading.style.display = "none";
    resultsEmpty.style.display = "flex";
    analyzeBtn.disabled = false;
    alert(`Prediction Failed: ${error.message}`);
  }
}

// ==========================================
// 6. RENDER PREDICTION RESULTS
// ==========================================
function displayPredictionResults(data) {
  resultsLoading.style.display = "none";
  resultsPopulated.style.display = "block";
  analyzeBtn.disabled = false;

  // 1. Demo Mode Notification
  const demoBanner = document.getElementById("demoNoticeBanner");
  if (data.is_demo_mode) {
    demoBanner.style.display = "flex";
    document.getElementById("demoNoticeText").textContent = data.demo_notice || "Simulated demo result.";
  } else {
    demoBanner.style.display = "none";
  }

  // 2. Low Confidence Caution
  const cautionBanner = document.getElementById("cautionBanner");
  if (data.is_low_confidence) {
    cautionBanner.style.display = "flex";
    document.getElementById("cautionText").textContent =
      `Model confidence is ${data.confidence_pct}% (below 60% threshold). This leaf might exhibit multiple overlapping disorders or be an unsupported plant species.`;
  } else {
    cautionBanner.style.display = "none";
  }

  // 3. Header titles & badge
  document.getElementById("resPlantType").textContent = data.plant_type;
  document.getElementById("resDiseaseName").textContent = data.disease_name;

  const badge = document.getElementById("resStatusBadge");
  badge.textContent = data.status_label;
  if (data.status === "healthy") {
    badge.className = "result-status-badge badge-healthy";
  } else if (data.status === "uncertain") {
    badge.className = "result-status-badge badge-uncertain";
  } else {
    badge.className = "result-status-badge badge-diseased";
  }

  // 4. Confidence Gauge
  document.getElementById("resConfidenceVal").textContent = `${data.confidence_pct}%`;
  document.getElementById("resConfidenceBar").style.width = `${Math.min(data.confidence_pct, 100)}%`;

  const rec = data.recommendations || {};
  document.getElementById("resSeverity").textContent = `Severity: ${rec.severity || "Unspecified"}`;

  // 5. Alternative Predictions
  const altContainer = document.getElementById("resTopAlternatives");
  altContainer.innerHTML = "";
  if (data.top_predictions && data.top_predictions.length > 0) {
    data.top_predictions.forEach(p => {
      const item = document.createElement("div");
      item.className = "alt-item";
      item.innerHTML = `
        <span><strong>${p.plant}</strong>: ${p.disease}</span>
        <span>${p.confidence_pct}%</span>
      `;
      altContainer.appendChild(item);
    });
  }

  // 6. Structured Diagnostic Sections
  populateList("resSymptomsList", rec.symptoms || ["No specific symptoms recorded."]);
  populateList("resCausesList", rec.causes || ["Environmental triggers vary."]);
  populateList("resImmediateList", rec.immediate_actions || ["Monitor leaf progress."]);
  populateList("resPreventionList", rec.preventative_measures || ["Follow good agricultural sanitation."]);
  populateList("resOrganicList", rec.organic_controls || ["Apply non-chemical cultural methods."]);

  document.getElementById("resExpertText").textContent =
    rec.expert_consultation || "Contact your local agricultural extension service for on-site verification.";

  document.getElementById("resDisclaimer").textContent =
    data.disclaimer || "AI predictions are assistive decision-support tools and are not a substitute for certified agricultural diagnosis.";

  // Scroll smoothly to results on mobile devices
  if (window.innerWidth < 992) {
    resultsPopulated.scrollIntoView({ behavior: "smooth" });
  }
}

function populateList(elementId, items) {
  const container = document.getElementById(elementId);
  container.innerHTML = "";
  items.forEach(text => {
    const li = document.createElement("li");
    li.textContent = text;
    container.appendChild(li);
  });
}

function resetDetection() {
  selectedImageFile = null;
  leafInput.value = "";
  previewImg.src = "";
  dropzonePreview.style.display = "none";
  dropzoneEmpty.style.display = "block";
  analyzeBtn.disabled = true;

  resultsPopulated.style.display = "none";
  resultsLoading.style.display = "none";
  resultsEmpty.style.display = "flex";
}

// ==========================================
// 7. DASHBOARD DATA & REAL-TIME STATS
// ==========================================
async function fetchDashboardData() {
  try {
    const res = await fetch(`${API_BASE}/api/history?limit=20`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const stats = data.statistics || {};
    document.getElementById("kpiTotalScans").textContent = stats.total_scans || 0;
    document.getElementById("kpiHealthy").textContent = stats.healthy_count || 0;
    document.getElementById("kpiDiseased").textContent = stats.diseased_count || 0;
    document.getElementById("kpiAvgConf").textContent = `${stats.average_confidence || 0}%`;

    // Crop Health Ratio
    const total = stats.total_scans || 0;
    if (total > 0) {
      const hPct = Math.round((stats.healthy_count / total) * 100);
      const dPct = 100 - hPct;
      document.getElementById("ratioHealthyBar").style.width = `${hPct}%`;
      document.getElementById("ratioDiseasedBar").style.width = `${dPct}%`;
      document.getElementById("ratioHealthyPct").textContent = `${hPct}%`;
      document.getElementById("ratioDiseasedPct").textContent = `${dPct}%`;
    } else {
      document.getElementById("ratioHealthyBar").style.width = "50%";
      document.getElementById("ratioDiseasedBar").style.width = "50%";
      document.getElementById("ratioHealthyPct").textContent = "0%";
      document.getElementById("ratioDiseasedPct").textContent = "0%";
    }

    // Top Detected Pathogens Bars
    const distContainer = document.getElementById("distributionBars");
    distContainer.innerHTML = "";
    const dist = stats.disease_distribution || [];
    if (dist.length === 0) {
      distContainer.innerHTML = `<div class="empty-state-sm">No disease incidents logged yet.</div>`;
    } else {
      const maxCount = Math.max(...dist.map(d => d.count), 1);
      dist.forEach(d => {
        const pct = Math.round((d.count / maxCount) * 100);
        const item = document.createElement("div");
        item.className = "dist-bar-item";
        item.innerHTML = `
          <div class="dist-bar-header">
            <span>${d.plant_type}: ${d.disease_name}</span>
            <span>${d.count} scan${d.count > 1 ? "s" : ""}</span>
          </div>
          <div class="dist-bar-track">
            <div class="dist-bar-progress" style="width: ${pct}%;"></div>
          </div>
        `;
        distContainer.appendChild(item);
      });
    }

    // History Table
    const tbody = document.getElementById("historyTableBody");
    tbody.innerHTML = "";
    const scans = data.recent_scans || [];

    if (scans.length === 0) {
      tbody.innerHTML = `
        <tr>
          <td colspan="7" class="empty-table-cell">No scan history recorded yet. Run your first leaf scan!</td>
        </tr>
      `;
    } else {
      scans.forEach(s => {
        const tr = document.createElement("tr");
        const statusBadgeClass = s.status === "healthy" 
          ? "badge-healthy" 
          : (s.is_low_confidence ? "badge-uncertain" : "badge-diseased");
        const statusText = s.is_low_confidence 
          ? "Uncertain" 
          : (s.status.charAt(0).toUpperCase() + s.status.slice(1));

        tr.innerHTML = `
          <td>#${s.id}</td>
          <td>${formatDate(s.timestamp)}</td>
          <td><strong>${s.plant_type}</strong></td>
          <td>${s.disease_name}</td>
          <td><span class="result-status-badge ${statusBadgeClass}" style="padding: 0.2rem 0.5rem; font-size: 0.75rem;">${statusText}</span></td>
          <td>${Math.round(s.confidence * 100)}%</td>
          <td style="color: #6b7280; font-size: 0.8rem;">${s.filename}</td>
        `;
        tbody.appendChild(tr);
      });
    }

  } catch (err) {
    console.error("Dashboard fetch error:", err);
  }
}

function formatDate(isoStr) {
  if (!isoStr) return "-";
  try {
    const d = new Date(isoStr);
    return d.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });
  } catch (e) {
    return isoStr;
  }
}

// Clear History Flow
function confirmClearHistory() {
  document.getElementById("clearModal").style.display = "flex";
}

function closeClearModal() {
  document.getElementById("clearModal").style.display = "none";
}

async function executeClearHistory() {
  try {
    const res = await fetch(`${API_BASE}/api/history/clear`, {
      method: "POST"
    });
    if (!res.ok) throw new Error("Failed to clear history");
    closeClearModal();
    fetchDashboardData();
  } catch (err) {
    alert(`Error: ${err.message}`);
  }
}

// ==========================================
// 8. DISEASE LIBRARY BROWSER
// ==========================================
async function fetchDiseaseLibrary() {
  const grid = document.getElementById("libraryGrid");
  try {
    const res = await fetch(`${API_BASE}/api/library`);
    if (!res.ok) throw new Error("Could not load library");
    const data = await res.json();
    diseaseLibraryCache = data.diseases || [];
    renderDiseaseLibrary(diseaseLibraryCache);
  } catch (err) {
    grid.innerHTML = `<div class="empty-state-sm">Failed to load disease library from server.</div>`;
  }
}

function renderDiseaseLibrary(items) {
  const grid = document.getElementById("libraryGrid");
  grid.innerHTML = "";

  if (items.length === 0) {
    grid.innerHTML = `<div class="empty-state-sm">No plant diseases matched your query.</div>`;
    return;
  }

  items.forEach(item => {
    const card = document.createElement("div");
    card.className = "lib-card";

    const badgeClass = item.is_healthy ? "badge-healthy" : "badge-diseased";
    const statusLabel = item.is_healthy ? "Healthy Foliage" : (item.severity || "Diseased");

    card.innerHTML = `
      <div>
        <div class="lib-top">
          <span class="lib-plant-tag">${item.plant}</span>
          <span class="result-status-badge ${badgeClass}" style="padding: 0.15rem 0.5rem; font-size: 0.72rem;">${statusLabel}</span>
        </div>
        <h3 class="lib-title">${item.disease}</h3>
        <p class="lib-desc">${item.description || "Agricultural disease profile."}</p>
        
        <div class="lib-meta">
          <strong>Key Symptoms:</strong>
          <span>${item.symptoms ? item.symptoms[0] : "Visual foliar signs."}</span>
        </div>

        <div class="lib-meta">
          <strong>Primary Prevention:</strong>
          <span>${item.preventative_measures ? item.preventative_measures[0] : "Proper air flow and clean seeds."}</span>
        </div>
      </div>
      
      <div style="margin-top: 1rem;">
        <button class="btn btn-outline btn-block btn-sm" onclick="prefillDetector('${item.plant}')">
          Scan ${item.plant} Leaf
        </button>
      </div>
    `;
    grid.appendChild(card);
  });
}

function filterLibrary() {
  const query = document.getElementById("librarySearch").value.toLowerCase().trim();

  const filtered = diseaseLibraryCache.filter(item => {
    const matchesCrop = activeCropFilter === "all" || item.plant.toLowerCase() === activeCropFilter.toLowerCase();
    const matchesSearch = query === "" ||
      item.plant.toLowerCase().includes(query) ||
      item.disease.toLowerCase().includes(query) ||
      (item.description && item.description.toLowerCase().includes(query));

    return matchesCrop && matchesSearch;
  });

  renderDiseaseLibrary(filtered);
}

function filterByCrop(cropName) {
  activeCropFilter = cropName;
  document.querySelectorAll(".filter-pill").forEach(p => {
    p.classList.toggle("active", p.getAttribute("data-filter") === cropName);
  });
  filterLibrary();
}

function prefillDetector(plantName) {
  navigateTo("detector");
}

// ==========================================
// 9. INITIALIZATION
// ==========================================
document.addEventListener("DOMContentLoaded", () => {
  checkSystemHealth();
  setupSampleChips();

  // Listen for browser popstate or hash changes if any
  const hash = window.location.hash.replace("#", "");
  if (["home", "detector", "dashboard", "library", "about"].includes(hash)) {
    navigateTo(hash);
  }
});
