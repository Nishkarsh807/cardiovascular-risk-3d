/**
 * CardioVision 3D - Anatomical Heart & Coronary Artery Visualizer
 * Three.js WebGL Engine with Dynamic Stenosis Risk Mapping
 */

class Heart3DVisualizer {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    if (!this.container) {
      console.error(`Container #${containerId} not found.`);
      return;
    }

    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.controls = null;
    this.raycaster = new THREE.Raycaster();
    this.mouse = new THREE.Vector2();

    // Anatomical Meshes & Groups
    this.heartGroup = new THREE.Group();
    this.vesselsGroup = new THREE.Group();
    this.torsoGroup = new THREE.Group();
    this.particleSystems = {};

    // Vessel Meshes & Materials
    this.vesselMeshes = {};
    this.vesselMaterials = {};
    this.vesselCurves = {};

    // Animation & State
    this.clock = new THREE.Clock();
    this.heartbeatEnabled = true;
    this.torsoVisible = true;
    this.bloodFlowEnabled = true;
    this.isTransitioningCamera = false;
    this.targetCameraPos = null;
    this.targetLookAt = null;

    // Current Risk State (Default: Normal)
    this.vesselRisks = {
      LAD: { probability: 0.15, color: 0x00e676 },
      LCX: { probability: 0.18, color: 0x00e676 },
      RCA: { probability: 0.12, color: 0x00e676 }
    };

    this.init();
  }

  init() {
    const width = this.container.clientWidth || window.innerWidth * 0.55;
    const height = this.container.clientHeight || window.innerHeight - 80;

    // 1. Scene
    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.FogExp2(0x0a0e17, 0.035);

    // 2. Camera
    this.camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    this.camera.position.set(0, 1.2, 9.5);

    // 3. Renderer
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.2;
    this.container.appendChild(this.renderer.domElement);

    // 4. OrbitControls
    this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.minDistance = 3.5;
    this.controls.maxDistance = 18.0;
    this.controls.target.set(0, 0, 0);

    // 5. Lighting
    this.setupLighting();

    // 6. Build Anatomy
    this.buildHeartAnatomy();
    this.buildCoronaryArteries();
    this.buildBloodFlowParticles();
    this.buildTorsoContext();

    this.scene.add(this.heartGroup);
    this.scene.add(this.torsoGroup);

    // 7. Event Listeners
    window.addEventListener("resize", () => this.onWindowResize());
    this.renderer.domElement.addEventListener("mousemove", (e) => this.onMouseMove(e));
    this.renderer.domElement.addEventListener("click", (e) => this.onClick(e));

    // 8. Animation Loop
    this.animate();
  }

  setupLighting() {
    // Ambient fill
    const ambientLight = new THREE.AmbientLight(0x223355, 1.2);
    this.scene.add(ambientLight);

    // Key surgical light
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
    keyLight.position.set(5, 8, 7);
    this.scene.add(keyLight);

    // Rim back light (cyan surgical rim)
    const rimLight = new THREE.DirectionalLight(0x00e5ff, 1.4);
    rimLight.position.set(-6, -4, -6);
    this.scene.add(rimLight);

    // Bottom soft reflection
    const bottomLight = new THREE.DirectionalLight(0x992244, 0.8);
    bottomLight.position.set(0, -6, 4);
    this.scene.add(bottomLight);
  }

  buildHeartAnatomy() {
    // Muscle Material
    const muscleMaterial = new THREE.MeshStandardMaterial({
      color: 0x8b1e2a,
      roughness: 0.45,
      metalness: 0.15,
      emissive: 0x220508,
      emissiveIntensity: 0.3
    });

    // Ventricles Main Body (Conical organic mass tilted towards apex)
    const ventricleGeo = new THREE.ConeGeometry(1.6, 2.8, 32);
    ventricleGeo.rotateX(Math.PI);
    ventricleGeo.translate(0, -0.6, 0);
    const ventricles = new THREE.Mesh(ventricleGeo, muscleMaterial);
    ventricles.scale.set(1.15, 1.1, 0.95);
    ventricles.rotation.z = 0.22;
    ventricles.userData = { name: "Ventricular Myocardium", role: "Primary pumping chambers (LV & RV)" };
    this.heartGroup.add(ventricles);

    // Left Ventricle Lateral Bulge
    const lvBulgeGeo = new THREE.SphereGeometry(1.3, 24, 24);
    lvBulgeGeo.scale(1.1, 1.3, 0.85);
    const lvBulge = new THREE.Mesh(lvBulgeGeo, muscleMaterial);
    lvBulge.position.set(-0.4, -0.4, -0.1);
    lvBulge.userData = { name: "Left Ventricle", role: "Systemic circulation pump" };
    this.heartGroup.add(lvBulge);

    // Right Ventricle Anterior Bulge
    const rvBulgeGeo = new THREE.SphereGeometry(1.1, 24, 24);
    rvBulgeGeo.scale(1.0, 1.1, 0.9);
    const rvBulge = new THREE.Mesh(rvBulgeGeo, muscleMaterial);
    rvBulge.position.set(0.65, -0.35, 0.3);
    rvBulge.userData = { name: "Right Ventricle", role: "Pulmonary circulation pump" };
    this.heartGroup.add(rvBulge);

    // Atria Base
    const atriaMaterial = new THREE.MeshStandardMaterial({
      color: 0x701824,
      roughness: 0.5,
      metalness: 0.1
    });

    const laGeo = new THREE.SphereGeometry(0.9, 20, 20);
    const laMesh = new THREE.Mesh(laGeo, atriaMaterial);
    laMesh.position.set(-0.7, 1.1, -0.3);
    laMesh.userData = { name: "Left Atrium", role: "Receives oxygenated blood from pulmonary veins" };
    this.heartGroup.add(laMesh);

    const raGeo = new THREE.SphereGeometry(0.95, 20, 20);
    const raMesh = new THREE.Mesh(raGeo, atriaMaterial);
    raMesh.position.set(0.9, 1.0, 0.1);
    raMesh.userData = { name: "Right Atrium", role: "Receives deoxygenated blood from vena cava" };
    this.heartGroup.add(raMesh);

    // Aorta & Aortic Arch (Bright Crimson / Ruby artery)
    const aortaMaterial = new THREE.MeshStandardMaterial({
      color: 0xcc2936,
      roughness: 0.3,
      metalness: 0.25,
      emissive: 0x440810,
      emissiveIntensity: 0.4
    });

    const aortaCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(-0.2, 0.8, 0.1),
      new THREE.Vector3(-0.1, 1.8, 0.2),
      new THREE.Vector3(0.0, 2.5, 0.0),
      new THREE.Vector3(-0.5, 2.6, -0.5),
      new THREE.Vector3(-0.8, 1.6, -0.8),
      new THREE.Vector3(-0.7, 0.4, -0.9)
    ]);
    const aortaGeo = new THREE.TubeGeometry(aortaCurve, 40, 0.36, 16, false);
    const aortaMesh = new THREE.Mesh(aortaGeo, aortaMaterial);
    aortaMesh.userData = { name: "Ascending Aorta & Arch", role: "Main systemic arterial trunk" };
    this.heartGroup.add(aortaMesh);

    // Brachiocephalic, Carotid, Subclavian Branches on Arch
    const branchGeo = new THREE.CylinderGeometry(0.09, 0.09, 0.7, 12);
    const branch1 = new THREE.Mesh(branchGeo, aortaMaterial);
    branch1.position.set(-0.05, 2.8, 0.1);
    branch1.rotation.z = -0.2;
    this.heartGroup.add(branch1);

    const branch2 = new THREE.Mesh(branchGeo, aortaMaterial);
    branch2.position.set(-0.3, 2.85, -0.15);
    branch2.rotation.z = -0.1;
    this.heartGroup.add(branch2);

    const branch3 = new THREE.Mesh(branchGeo, aortaMaterial);
    branch3.position.set(-0.55, 2.8, -0.4);
    branch3.rotation.z = 0.15;
    this.heartGroup.add(branch3);

    // Pulmonary Artery (Blue-Violet venous/pulmonary trunk)
    const pulmonaryMaterial = new THREE.MeshStandardMaterial({
      color: 0x3a5a9f,
      roughness: 0.35,
      metalness: 0.2
    });

    const pulmonaryCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.3, 0.9, 0.5),
      new THREE.Vector3(0.1, 1.7, 0.4),
      new THREE.Vector3(-0.2, 2.0, 0.1),
      new THREE.Vector3(-0.6, 2.1, -0.2)
    ]);
    const pulmonaryGeo = new THREE.TubeGeometry(pulmonaryCurve, 30, 0.32, 16, false);
    const pulmonaryMesh = new THREE.Mesh(pulmonaryGeo, pulmonaryMaterial);
    pulmonaryMesh.userData = { name: "Pulmonary Trunk", role: "Carries deoxygenated blood to lungs" };
    this.heartGroup.add(pulmonaryMesh);

    // Superior Vena Cava (SVC)
    const svcGeo = new THREE.CylinderGeometry(0.25, 0.25, 1.4, 16);
    const svcMesh = new THREE.Mesh(svcGeo, pulmonaryMaterial);
    svcMesh.position.set(1.15, 1.9, -0.2);
    svcMesh.userData = { name: "Superior Vena Cava", role: "Returns systemic blood from upper body" };
    this.heartGroup.add(svcMesh);
  }

  buildCoronaryArteries() {
    // 1. LAD (Left Anterior Descending)
    // Courses from aortic root down the anterior interventricular sulcus to the apex
    const ladPoints = [
      new THREE.Vector3(-0.25, 0.95, 0.45),
      new THREE.Vector3(-0.15, 0.65, 0.8),
      new THREE.Vector3(0.05, 0.25, 1.05),
      new THREE.Vector3(0.15, -0.3, 1.15),
      new THREE.Vector3(0.18, -0.9, 1.08),
      new THREE.Vector3(0.12, -1.5, 0.85),
      new THREE.Vector3(0.05, -2.0, 0.45) // Near Apex
    ];
    this.vesselCurves["LAD"] = new THREE.CatmullRomCurve3(ladPoints);

    // LAD Diagonal Branch (D1)
    const ladD1Points = [
      new THREE.Vector3(0.05, 0.25, 1.05),
      new THREE.Vector3(-0.35, -0.15, 0.95),
      new THREE.Vector3(-0.7, -0.65, 0.7)
    ];
    this.ladD1Curve = new THREE.CatmullRomCurve3(ladD1Points);

    // 2. LCX (Left Circumflex Artery)
    // Branches from left main, wraps around in the left AV groove to posterior wall
    const lcxPoints = [
      new THREE.Vector3(-0.25, 0.95, 0.45),
      new THREE.Vector3(-0.75, 0.7, 0.3),
      new THREE.Vector3(-1.25, 0.45, 0.0),
      new THREE.Vector3(-1.45, 0.15, -0.45),
      new THREE.Vector3(-1.3, -0.35, -0.85),
      new THREE.Vector3(-0.85, -0.9, -1.05),
      new THREE.Vector3(-0.35, -1.35, -1.0)
    ];
    this.vesselCurves["LCX"] = new THREE.CatmullRomCurve3(lcxPoints);

    // LCX Obtuse Marginal Branch (OM1)
    const lcxOMPoints = [
      new THREE.Vector3(-1.25, 0.45, 0.0),
      new THREE.Vector3(-1.35, -0.2, 0.15),
      new THREE.Vector3(-1.2, -0.8, 0.2)
    ];
    this.lcxOMCurve = new THREE.CatmullRomCurve3(lcxOMPoints);

    // 3. RCA (Right Coronary Artery)
    // Emerges from right aortic sinus, descends through right AV sulcus to posterior descending artery
    const rcaPoints = [
      new THREE.Vector3(0.15, 0.9, 0.4),
      new THREE.Vector3(0.7, 0.7, 0.55),
      new THREE.Vector3(1.15, 0.35, 0.5),
      new THREE.Vector3(1.25, -0.2, 0.35),
      new THREE.Vector3(1.15, -0.8, 0.0),
      new THREE.Vector3(0.85, -1.2, -0.4),
      new THREE.Vector3(0.4, -1.55, -0.75),
      new THREE.Vector3(0.15, -1.9, -0.5) // PDA
    ];
    this.vesselCurves["RCA"] = new THREE.CatmullRomCurve3(rcaPoints);

    // Create Tubes and Materials for LAD, LCX, RCA
    ["LAD", "LCX", "RCA"].forEach((key) => {
      const mat = new THREE.MeshStandardMaterial({
        color: this.vesselRisks[key].color,
        roughness: 0.25,
        metalness: 0.4,
        emissive: this.vesselRisks[key].color,
        emissiveIntensity: 0.6
      });
      this.vesselMaterials[key] = mat;

      const geo = new THREE.TubeGeometry(this.vesselCurves[key], 64, 0.085, 12, false);
      const mesh = new THREE.Mesh(geo, mat);
      mesh.userData = {
        isVessel: true,
        vesselKey: key,
        name: key === "LAD" ? "Left Anterior Descending (LAD)" : key === "LCX" ? "Left Circumflex (LCX)" : "Right Coronary Artery (RCA)"
      };

      this.vesselMeshes[key] = mesh;
      this.vesselsGroup.add(mesh);
    });

    // Add branch tubes (child segments sharing vessel color)
    const d1Geo = new THREE.TubeGeometry(this.ladD1Curve, 30, 0.055, 10, false);
    const d1Mesh = new THREE.Mesh(d1Geo, this.vesselMaterials["LAD"]);
    d1Mesh.userData = { isVessel: true, vesselKey: "LAD", name: "LAD Diagonal Branch (D1)" };
    this.vesselsGroup.add(d1Mesh);

    const omGeo = new THREE.TubeGeometry(this.lcxOMCurve, 30, 0.055, 10, false);
    const omMesh = new THREE.Mesh(omGeo, this.vesselMaterials["LCX"]);
    omMesh.userData = { isVessel: true, vesselKey: "LCX", name: "LCX Obtuse Marginal Branch (OM1)" };
    this.vesselsGroup.add(omMesh);

    this.heartGroup.add(this.vesselsGroup);
  }

  buildBloodFlowParticles() {
    // Hemodynamic animated particles traveling through coronary arteries
    const particleCountPerVessel = 60;
    const particleMat = new THREE.PointsMaterial({
      color: 0xffffff,
      size: 0.075,
      transparent: true,
      opacity: 0.85,
      blending: THREE.AdditiveBlending
    });

    ["LAD", "LCX", "RCA"].forEach((key) => {
      const geo = new THREE.BufferGeometry();
      const positions = new Float32Array(particleCountPerVessel * 3);
      const offsets = new Float32Array(particleCountPerVessel);

      for (let i = 0; i < particleCountPerVessel; i++) {
        offsets[i] = i / particleCountPerVessel;
        const pt = this.vesselCurves[key].getPoint(offsets[i]);
        positions[i * 3] = pt.x;
        positions[i * 3 + 1] = pt.y;
        positions[i * 3 + 2] = pt.z;
      }

      geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
      const points = new THREE.Points(geo, particleMat.clone());
      this.particleSystems[key] = { points, offsets, curve: this.vesselCurves[key] };
      this.vesselsGroup.add(points);
    });
  }

  buildTorsoContext() {
    // Holographic Thoracic / Ribcage Outline for Anatomical Context
    const ribMaterial = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      wireframe: true,
      transparent: true,
      opacity: 0.12
    });

    // Rib cage ellipses
    for (let i = 0; i < 7; i++) {
      const y = -1.8 + i * 0.7;
      const radiusX = 2.8 + Math.sin(i * 0.45) * 0.9;
      const radiusZ = 2.2 + Math.sin(i * 0.45) * 0.6;
      const curve = new THREE.EllipseCurve(0, 0, radiusX, radiusZ, 0, Math.PI * 2, false, 0);
      const points = curve.getPoints(50);
      const ribGeo = new THREE.BufferGeometry().setFromPoints(
        points.map((p) => new THREE.Vector3(p.x, y, p.y))
      );
      const ribLine = new THREE.Line(ribGeo, ribMaterial);
      this.torsoGroup.add(ribLine);
    }

    // Sternum front strut
    const sternumGeo = new THREE.BoxGeometry(0.35, 4.5, 0.1);
    const sternumMesh = new THREE.Mesh(sternumGeo, ribMaterial);
    sternumMesh.position.set(0, 0.4, 2.5);
    this.torsoGroup.add(sternumMesh);

    // Spine back strut
    const spineGeo = new THREE.CylinderGeometry(0.2, 0.2, 5.0, 16);
    const spineMesh = new THREE.Mesh(spineGeo, ribMaterial);
    spineMesh.position.set(0, 0.2, -2.3);
    this.torsoGroup.add(spineMesh);
  }

  updateVesselColors(vesselData) {
    /**
     * Dynamically updates the color-code and glow of LAD, LCX, RCA
     * based on predicted stenosis probabilities.
     */
    ["LAD", "LCX", "RCA"].forEach((key) => {
      if (!vesselData || !vesselData[key]) return;

      const prob = vesselData[key].probability;
      const colorHex = vesselData[key].color_code || (prob >= 0.60 ? "#ff1744" : prob >= 0.35 ? "#ffb300" : "#00e676");
      const threeColor = new THREE.Color(colorHex);

      this.vesselRisks[key] = { probability: prob, color: threeColor };

      if (this.vesselMaterials[key]) {
        this.vesselMaterials[key].color.set(threeColor);
        this.vesselMaterials[key].emissive.set(threeColor);
        // High risk vessels emit stronger pulsating glow
        this.vesselMaterials[key].emissiveIntensity = prob >= 0.60 ? 1.0 : prob >= 0.35 ? 0.6 : 0.4;
      }

      // Update particle blood flow speed & color
      if (this.particleSystems[key]) {
        this.particleSystems[key].points.material.color.set(threeColor);
      }
    });
  }

  focusVessel(vesselKey) {
    /** Smoothly orbits and centers camera on the selected coronary vessel */
    this.isTransitioningCamera = true;
    if (vesselKey === "LAD") {
      this.targetCameraPos = new THREE.Vector3(0.2, -0.2, 6.2);
      this.targetLookAt = new THREE.Vector3(0.1, -0.5, 0.6);
    } else if (vesselKey === "LCX") {
      this.targetCameraPos = new THREE.Vector3(-4.8, 0.4, -2.8);
      this.targetLookAt = new THREE.Vector3(-0.8, -0.2, -0.4);
    } else if (vesselKey === "RCA") {
      this.targetCameraPos = new THREE.Vector3(4.8, -0.3, 3.2);
      this.targetLookAt = new THREE.Vector3(0.8, -0.4, 0.2);
    }
  }

  focusAnterior() {
    this.isTransitioningCamera = true;
    this.targetCameraPos = new THREE.Vector3(0, 0.6, 9.0);
    this.targetLookAt = new THREE.Vector3(0, 0, 0);
  }

  focusPosterior() {
    this.isTransitioningCamera = true;
    this.targetCameraPos = new THREE.Vector3(0, 0.6, -9.0);
    this.targetLookAt = new THREE.Vector3(0, 0, 0);
  }

  resetView() {
    this.isTransitioningCamera = true;
    this.targetCameraPos = new THREE.Vector3(0, 1.2, 9.5);
    this.targetLookAt = new THREE.Vector3(0, 0, 0);
  }

  toggleHeartbeat() {
    this.heartbeatEnabled = !this.heartbeatEnabled;
    return this.heartbeatEnabled;
  }

  toggleTorso() {
    this.torsoVisible = !this.torsoVisible;
    this.torsoGroup.visible = this.torsoVisible;
    return this.torsoVisible;
  }

  toggleBloodFlow() {
    this.bloodFlowEnabled = !this.bloodFlowEnabled;
    Object.values(this.particleSystems).forEach((sys) => {
      sys.points.visible = this.bloodFlowEnabled;
    });
    return this.bloodFlowEnabled;
  }

  onMouseMove(event) {
    const rect = this.renderer.domElement.getBoundingClientRect();
    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(this.vesselsGroup.children, true);

    const tooltip = document.getElementById("hud-vessel-tooltip");
    if (intersects.length > 0) {
      const hit = intersects[0].object;
      if (hit.userData && hit.userData.isVessel) {
        document.body.style.cursor = "pointer";
        if (tooltip) {
          const key = hit.userData.vesselKey;
          const riskInfo = this.vesselRisks[key] || { probability: 0 };
          const probPct = (riskInfo.probability * 100).toFixed(1);
          tooltip.style.display = "block";
          tooltip.innerHTML = `
            <div class="tooltip-title">${hit.userData.name}</div>
            <div class="tooltip-desc">
              Predicted Stenosis Risk: <b style="color:${riskInfo.color ? '#' + riskInfo.color.getHexString() : '#fff'}">${probPct}%</b><br>
              Click vessel to focus camera & view SHAP attributions.
            </div>
          `;
        }
        return;
      }
    }
    document.body.style.cursor = "default";
    if (tooltip) tooltip.style.display = "none";
  }

  onClick(event) {
    this.raycaster.setFromCamera(this.mouse, this.camera);
    const intersects = this.raycaster.intersectObjects(this.vesselsGroup.children, true);

    if (intersects.length > 0) {
      const hit = intersects[0].object;
      if (hit.userData && hit.userData.vesselKey) {
        const key = hit.userData.vesselKey;
        this.focusVessel(key);
        // Highlight corresponding card in dashboard
        if (window.CardioApp && window.CardioApp.selectVesselCard) {
          window.CardioApp.selectVesselCard(key);
        }
      }
    }
  }

  onWindowResize() {
    if (!this.container || !this.renderer || !this.camera) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  animate() {
    requestAnimationFrame(() => this.animate());

    const delta = this.clock.getDelta();
    const elapsedTime = this.clock.getElapsedTime();

    // 1. Cardiac Cycle Heartbeat Pulse (Systole & Diastole contraction)
    if (this.heartbeatEnabled) {
      // Cardiac pulse equation: rapid systolic contraction followed by diastolic filling
      const cycle = (elapsedTime * 1.25) % 1.0;
      let pulse = 1.0;
      if (cycle < 0.15) {
        pulse = 1.0 + Math.sin((cycle / 0.15) * Math.PI) * 0.055; // First lub
      } else if (cycle < 0.35) {
        pulse = 1.0 + Math.sin(((cycle - 0.15) / 0.2) * Math.PI) * 0.035; // Second dub
      }
      this.heartGroup.scale.set(pulse, pulse, pulse);
    }

    // 2. High-Risk Vessel Warning Glow Pulsation
    ["LAD", "LCX", "RCA"].forEach((key) => {
      const risk = this.vesselRisks[key];
      if (risk && risk.probability >= 0.60 && this.vesselMaterials[key]) {
        // Red warning glow oscillation
        const glow = 0.7 + Math.sin(elapsedTime * 6) * 0.35;
        this.vesselMaterials[key].emissiveIntensity = glow;
      }
    });

    // 3. Hemodynamic Particle Blood Flow Animation
    if (this.bloodFlowEnabled) {
      ["LAD", "LCX", "RCA"].forEach((key) => {
        const sys = this.particleSystems[key];
        if (!sys) return;

        const prob = this.vesselRisks[key] ? this.vesselRisks[key].probability : 0.2;
        // Blood flow speed drops if artery is stenotic (simulates vascular resistance!)
        const speed = Math.max(0.04, (1.0 - prob * 0.75) * 0.25);

        const posAttr = sys.points.geometry.attributes.position;
        const count = sys.offsets.length;

        for (let i = 0; i < count; i++) {
          sys.offsets[i] = (sys.offsets[i] + delta * speed) % 1.0;
          const pt = sys.curve.getPoint(sys.offsets[i]);
          posAttr.setXYZ(i, pt.x, pt.y, pt.z);
        }
        posAttr.needsUpdate = true;
      });
    }

    // 4. Smooth Camera Transition (Lerp)
    if (this.isTransitioningCamera && this.targetCameraPos) {
      this.camera.position.lerp(this.targetCameraPos, 0.065);
      this.controls.target.lerp(this.targetLookAt, 0.065);

      if (this.camera.position.distanceTo(this.targetCameraPos) < 0.05) {
        this.isTransitioningCamera = false;
      }
    }

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }
}

window.Heart3DVisualizer = Heart3DVisualizer;
