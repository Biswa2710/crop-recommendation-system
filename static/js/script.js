/*
 * script.js
 * CropWise AI — Client-Side Controller & AJAX Engine
 */

document.addEventListener('DOMContentLoaded', () => {

    // --- PRESET DATA SETS ---
    const PRESETS = {
        rice:       { N: 90, P: 42, K: 43, temperature: 20.8, humidity: 82.0, ph: 6.5, rainfall: 202.9 },
        maize:      { N: 80, P: 48, K: 20, temperature: 22.6, humidity: 65.2, ph: 6.2, rainfall: 80.5 },
        cotton:     { N: 120, P: 45, K: 20, temperature: 24.5, humidity: 80.0, ph: 6.8, rainfall: 82.0 },
        coffee:     { N: 100, P: 28, K: 30, temperature: 25.4, humidity: 60.5, ph: 6.7, rainfall: 158.2 },
        apple:      { N: 20, P: 130, K: 200, temperature: 22.5, humidity: 92.0, ph: 6.0, rainfall: 110.0 },
        watermelon: { N: 100, P: 18, K: 50, temperature: 25.5, humidity: 85.0, ph: 6.4, rainfall: 50.0 }
    };

    // --- SUPPORTED CROPS LIST FOR GRID ---
    const SUPPORTED_CROPS = [
        { key: "rice", name: "Rice", icon: "🌾", category: "Cereal Grain", season: "Kharif" },
        { key: "maize", name: "Maize (Corn)", icon: "🌽", category: "Cereal Grain", season: "Kharif & Rabi" },
        { key: "chickpea", name: "Chickpea", icon: "🧆", category: "Pulse / Legume", season: "Rabi" },
        { key: "kidneybeans", name: "Kidney Beans", icon: "🫘", category: "Pulse / Legume", season: "Rabi" },
        { key: "pigeonpeas", name: "Pigeon Peas", icon: "🌱", category: "Pulse / Legume", season: "Kharif" },
        { key: "mothbeans", name: "Moth Beans", icon: "🌿", category: "Pulse / Legume", season: "Kharif" },
        { key: "mungbean", name: "Mung Bean", icon: "🌱", category: "Pulse / Legume", season: "Kharif & Summer" },
        { key: "blackgram", name: "Black Gram", icon: "🫘", category: "Pulse / Legume", season: "Kharif & Summer" },
        { key: "lentil", name: "Lentil", icon: "🍲", category: "Pulse / Legume", season: "Rabi" },
        { key: "pomegranate", name: "Pomegranate", icon: "🍎", category: "Fruit", season: "Perennial" },
        { key: "banana", name: "Banana", icon: "🍌", category: "Fruit", season: "Perennial" },
        { key: "mango", name: "Mango", icon: "🥭", category: "Fruit", season: "Perennial" },
        { key: "grapes", name: "Grapes", icon: "🍇", category: "Fruit", season: "Perennial" },
        { key: "watermelon", name: "Watermelon", icon: "🍉", category: "Fruit", season: "Summer / Zaid" },
        { key: "muskmelon", name: "Muskmelon", icon: "🍈", category: "Fruit", season: "Summer / Zaid" },
        { key: "apple", name: "Apple", icon: "🍎", category: "Fruit", season: "Temperate" },
        { key: "orange", name: "Orange", icon: "🍊", category: "Fruit", season: "Sub-tropical" },
        { key: "papaya", name: "Papaya", icon: "🥭", category: "Fruit", season: "Year-round" },
        { key: "coconut", name: "Coconut", icon: "🥥", category: "Plantation", season: "Perennial" },
        { key: "cotton", name: "Cotton", icon: "☁️", category: "Fiber Crop", season: "Kharif" },
        { key: "jute", name: "Jute", icon: "🎋", category: "Fiber Crop", season: "Kharif" },
        { key: "coffee", name: "Coffee", icon: "☕", category: "Commercial", season: "Perennial" }
    ];

    // --- CROP REQUIREMENTS SEARCH & AUTO-FILL ---
    const cropSearchInput = document.getElementById('crop-search-input');
    const cropSearchBtn = document.getElementById('crop-search-btn');
    const cropSearchResult = document.getElementById('crop-search-result');

    async function handleCropSearch() {
        if (!cropSearchInput || !cropSearchResult) return;
        const query = cropSearchInput.value.trim();
        if (!query) return;

        try {
            const response = await fetch(`/api/search-crop?q=${encodeURIComponent(query)}`);
            const data = await response.json();

            if (!response.ok || !data.success || !data.matches || data.matches.length === 0) {
                cropSearchResult.classList.remove('hidden');
                cropSearchResult.innerHTML = `
                    <div class="alert-box alert-danger">
                        ⚠️ No supported crops found matching "${query}". Try searching for <em>Rice, Coffee, Apple, Mango, Cotton, Watermelon</em>, etc.
                    </div>
                `;
                return;
            }

            const crop = data.matches[0]; // Take top match
            const params = crop.ideal_params || {};

            cropSearchResult.classList.remove('hidden');
            cropSearchResult.innerHTML = `
                <div class="search-crop-header">
                    <div class="search-crop-title">
                        <span class="search-crop-icon">${crop.icon}</span>
                        <div class="search-crop-meta">
                            <h4>${crop.name}</h4>
                            <span>${crop.category} • ${crop.season}</span>
                        </div>
                    </div>
                    <button type="button" class="btn btn-primary btn-sm autofill-btn" id="autofill-btn">
                        <i class="fa-solid fa-bolt"></i> Auto-Fill Predictor Form
                    </button>
                </div>

                <div class="search-ideal-grid">
                    <div class="ideal-param-item">
                        <div class="ideal-param-label">Ideal Nitrogen (N)</div>
                        <div class="ideal-param-value">${params.N || '-'} mg/kg</div>
                    </div>
                    <div class="ideal-param-item">
                        <div class="ideal-param-label">Ideal Phosphorus (P)</div>
                        <div class="ideal-param-value">${params.P || '-'} mg/kg</div>
                    </div>
                    <div class="ideal-param-item">
                        <div class="ideal-param-label">Ideal Potassium (K)</div>
                        <div class="ideal-param-value">${params.K || '-'} mg/kg</div>
                    </div>
                    <div class="ideal-param-item">
                        <div class="ideal-param-label">Ideal Temperature</div>
                        <div class="ideal-param-value">${params.temperature || '-'} °C</div>
                    </div>
                    <div class="ideal-param-item">
                        <div class="ideal-param-label">Ideal Humidity</div>
                        <div class="ideal-param-value">${params.humidity || '-'} %</div>
                    </div>
                    <div class="ideal-param-item">
                        <div class="ideal-param-label">Ideal Soil pH</div>
                        <div class="ideal-param-value">${params.ph || '-'} pH</div>
                    </div>
                    <div class="ideal-param-item">
                        <div class="ideal-param-label">Ideal Rainfall</div>
                        <div class="ideal-param-value">${params.rainfall || '-'} mm</div>
                    </div>
                    <div class="ideal-param-item">
                        <div class="ideal-param-label">Soil Type</div>
                        <div class="ideal-param-value" style="font-size:0.85rem; font-family:inherit;">${crop.soil_type || 'Fertile Soil'}</div>
                    </div>
                </div>

                <div class="agri-item" style="background: var(--gray-50); padding:14px; border-radius:8px;">
                    <div class="agri-item-title"><i class="fa-solid fa-lightbulb text-amber"></i> Cultivation Tip for ${crop.name}</div>
                    <p style="font-size:0.9rem; color:var(--dark-600);">${crop.farming_guidance}</p>
                </div>
            `;

            // Attach Auto-Fill button event
            const autofillBtn = document.getElementById('autofill-btn');
            if (autofillBtn && params.N !== undefined) {
                autofillBtn.addEventListener('click', () => {
                    document.getElementById('N').value = params.N;
                    document.getElementById('P').value = params.P;
                    document.getElementById('K').value = params.K;
                    document.getElementById('temperature').value = params.temperature;
                    document.getElementById('humidity').value = params.humidity;
                    document.getElementById('ph').value = params.ph;
                    document.getElementById('rainfall').value = params.rainfall;
                    hideAlert();
                    document.getElementById('prediction-form').scrollIntoView({ behavior: 'smooth', block: 'center' });
                });
            }

        } catch (e) {
            console.error("Crop search error:", e);
        }
    }

    if (cropSearchBtn) cropSearchBtn.addEventListener('click', handleCropSearch);
    if (cropSearchInput) {
        cropSearchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                handleCropSearch();
            }
        });
    }
    const form = document.getElementById('prediction-form');
    const submitBtn = document.getElementById('submit-btn');
    const alertBox = document.getElementById('alert-box');
    const resultWrapper = document.getElementById('result-wrapper');
    const cropSearch = document.getElementById('crop-search');
    const cropsGrid = document.getElementById('crops-grid');

    // --- POPULATE CROPS GRID ---
    function renderCropsGrid(crops) {
        if (!cropsGrid) return;
        cropsGrid.innerHTML = '';
        crops.forEach(crop => {
            const card = document.createElement('div');
            card.className = 'crop-card-item';
            card.innerHTML = `
                <div class="crop-card-icon">${crop.icon}</div>
                <div class="crop-card-title">${crop.name}</div>
                <div class="crop-card-category">${crop.category} • ${crop.season}</div>
            `;
            cropsGrid.appendChild(card);
        });
    }

    renderCropsGrid(SUPPORTED_CROPS);

    // --- CROP SEARCH FILTER ---
    if (cropSearch) {
        cropSearch.addEventListener('input', (e) => {
            const query = e.target.value.toLowerCase().trim();
            const filtered = SUPPORTED_CROPS.filter(c => 
                c.name.toLowerCase().includes(query) || 
                c.category.toLowerCase().includes(query) || 
                c.season.toLowerCase().includes(query)
            );
            renderCropsGrid(filtered);
        });
    }

    // --- PRESET BUTTON HANDLERS ---
    document.querySelectorAll('.preset-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const presetKey = btn.getAttribute('data-preset');
            if (PRESETS[presetKey]) {
                const data = PRESETS[presetKey];
                document.getElementById('N').value = data.N;
                document.getElementById('P').value = data.P;
                document.getElementById('K').value = data.K;
                document.getElementById('temperature').value = data.temperature;
                document.getElementById('humidity').value = data.humidity;
                document.getElementById('ph').value = data.ph;
                document.getElementById('rainfall').value = data.rainfall;
                hideAlert();
            }
        });
    });

    // --- FORM VALIDATION & SUBMIT HANDLER ---
    if (form) {
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            hideAlert();

            const N = parseFloat(document.getElementById('N').value);
            const P = parseFloat(document.getElementById('P').value);
            const K = parseFloat(document.getElementById('K').value);
            const temp = parseFloat(document.getElementById('temperature').value);
            const humidity = parseFloat(document.getElementById('humidity').value);
            const ph = parseFloat(document.getElementById('ph').value);
            const rainfall = parseFloat(document.getElementById('rainfall').value);

            // Client-side range validation
            if (isNaN(N) || N < 0 || N > 140) return showAlert("Nitrogen (N) must be between 0 and 140 mg/kg.");
            if (isNaN(P) || P < 5 || P > 145) return showAlert("Phosphorus (P) must be between 5 and 145 mg/kg.");
            if (isNaN(K) || K < 5 || K > 205) return showAlert("Potassium (K) must be between 5 and 205 mg/kg.");
            if (isNaN(temp) || temp < 8.0 || temp > 45.0) return showAlert("Temperature must be between 8.0 °C and 45.0 °C.");
            if (isNaN(humidity) || humidity < 14.0 || humidity > 100.0) return showAlert("Humidity must be between 14.0 % and 100.0 %.");
            if (isNaN(ph) || ph < 3.5 || ph > 10.0) return showAlert("Soil pH must be between 3.5 and 10.0.");
            if (isNaN(rainfall) || rainfall < 20.0 || rainfall > 350.0) return showAlert("Rainfall must be between 20.0 mm and 350.0 mm.");

            // Set loading state
            setLoading(true);

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ N, P, K, temperature: temp, humidity, ph, rainfall })
                });

                const data = await response.json();
                setLoading(false);

                if (!response.ok || !data.success) {
                    showAlert(data.error || "An error occurred while processing the prediction.");
                    return;
                }

                // Render result dynamically
                renderPredictionResult(data);

            } catch (err) {
                setLoading(false);
                showAlert("Network or server connection error. Please try again.");
            }
        });
    }

    function renderPredictionResult(data) {
        if (!resultWrapper) return;
        resultWrapper.classList.remove('hidden');

        // Update elements
        document.getElementById('res-crop-name').textContent = data.crop_name;
        document.querySelector('.crop-icon-large').textContent = data.crop_icon;
        document.getElementById('res-explanation').textContent = data.explanation;

        // Confidence
        const confVal = document.querySelector('.conf-value');
        const progressFill = document.querySelector('.progress-fill');
        if (confVal && data.confidence) {
            confVal.textContent = data.confidence + '%';
            if (progressFill) progressFill.style.width = data.confidence + '%';
        }

        // Alternatives
        const altContainer = document.querySelector('.alternatives-box');
        if (altContainer) {
            if (data.alternatives && data.alternatives.length > 0) {
                altContainer.classList.remove('hidden');
                const chips = altContainer.querySelector('.alt-chips');
                chips.innerHTML = data.alternatives.map(alt => `
                    <div class="alt-chip">
                        <span class="alt-icon">${alt.icon}</span>
                        <span class="alt-name">${alt.crop_name}</span>
                        <span class="alt-conf">${alt.confidence}%</span>
                    </div>
                `).join('');
            } else {
                altContainer.classList.add('hidden');
            }
        }

        // Summary Chips
        const summaryGrid = document.querySelector('.summary-chips-grid');
        if (summaryGrid && data.inputs) {
            summaryGrid.innerHTML = `
                <div class="chip"><span>Nitrogen (N):</span> <strong>${data.inputs.N} mg/kg</strong></div>
                <div class="chip"><span>Phosphorus (P):</span> <strong>${data.inputs.P} mg/kg</strong></div>
                <div class="chip"><span>Potassium (K):</span> <strong>${data.inputs.K} mg/kg</strong></div>
                <div class="chip"><span>Temperature:</span> <strong>${data.inputs.temperature} °C</strong></div>
                <div class="chip"><span>Humidity:</span> <strong>${data.inputs.humidity} %</strong></div>
                <div class="chip"><span>Soil pH:</span> <strong>${data.inputs.ph} pH</strong></div>
                <div class="chip"><span>Rainfall:</span> <strong>${data.inputs.rainfall} mm</strong></div>
            `;
        }

        // Agricultural Guidance Card
        if (data.crop_info) {
            document.getElementById('agri-soil').textContent = data.crop_info.soil_type || '-';
            document.getElementById('agri-season').textContent = data.crop_info.season || '-';
            document.getElementById('agri-npk').textContent = data.crop_info.npk_profile || '-';
            document.getElementById('agri-tips').textContent = data.crop_info.farming_guidance || '-';
        }

        // Scroll smoothly to results
        resultWrapper.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function showAlert(msg) {
        if (!alertBox) return;
        alertBox.textContent = "⚠️ " + msg;
        alertBox.className = "alert-box alert-danger";
        alertBox.classList.remove('hidden');
    }

    function hideAlert() {
        if (!alertBox) return;
        alertBox.classList.add('hidden');
    }

    function setLoading(isLoading) {
        if (!submitBtn) return;
        const textSpan = submitBtn.querySelector('.btn-text');
        const loaderSpan = submitBtn.querySelector('.btn-loader');
        if (isLoading) {
            submitBtn.disabled = true;
            if (textSpan) textSpan.classList.add('hidden');
            if (loaderSpan) loaderSpan.classList.remove('hidden');
        } else {
            submitBtn.disabled = false;
            if (textSpan) textSpan.classList.remove('hidden');
            if (loaderSpan) loaderSpan.classList.add('hidden');
        }
    }

    // --- FETCH ML MODEL INFORMATION FOR TABLE ---
    async function loadModelMetadata() {
        try {
            const res = await fetch('/api/model-info');
            if (!res.ok) return;
            const data = await res.json();

            if (data.best_model_name) {
                const nameEl = document.getElementById('model-best-name');
                const accEl = document.getElementById('model-best-acc');
                const f1El = document.getElementById('model-best-f1');
                if (nameEl) nameEl.textContent = data.best_model_name;
                if (accEl) accEl.textContent = (data.best_accuracy * 100).toFixed(2) + '%';
                if (f1El) f1El.textContent = (data.best_f1_weighted * 100).toFixed(2) + '%';
            }

            if (data.model_comparison && data.model_comparison.length > 0) {
                const tableBody = document.getElementById('model-table-body');
                if (tableBody) {
                    tableBody.innerHTML = data.model_comparison.map(m => {
                        const isBest = m.model_name === data.best_model_name;
                        return `
                            <tr>
                                <td>${isBest ? `<strong>${m.model_name} (Selected)</strong>` : m.model_name}</td>
                                <td>${isBest ? `<span class="badge badge-success">${(m.accuracy * 100).toFixed(2)}%</span>` : `${(m.accuracy * 100).toFixed(2)}%`}</td>
                                <td>${(m.f1_weighted * 100).toFixed(2)}%</td>
                            </tr>
                        `;
                    }).join('');
                }
            }
        } catch (e) {
            console.log("Could not load dynamic model metadata:", e);
        }
    }

    loadModelMetadata();
});
