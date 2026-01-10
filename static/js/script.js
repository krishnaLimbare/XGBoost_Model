document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const resultContainer = document.getElementById('resultContainer');
    const resetBtn = document.getElementById('resetBtn');
    const predictBtn = document.getElementById('predictBtn');

    // Elements to update
    const resultTitle = document.getElementById('resultTitle');
    const resultIcon = document.getElementById('resultIcon');
    const meterFill = document.getElementById('meterFill');
    const probabilityText = document.getElementById('probabilityText');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Show loading state
        predictBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
        predictBtn.disabled = true;

        // Gather data
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        // Convert number strings to numbers
        data.age = parseFloat(data.age);
        data.bmi = parseFloat(data.bmi);
        data.hypertension = parseInt(data.hypertension);
        data.heart_disease = parseInt(data.heart_disease);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            });

            const result = await response.json();

            if (result.status === 'success') {
                showResult(result);
            } else {
                alert('Error: ' + result.message);
            }

        } catch (error) {
            console.error('Error:', error);
            alert('An error occurred during prediction.');
        } finally {
            predictBtn.innerHTML = '<span>Analyze Risk</span> <i class="fa-solid fa-arrow-right"></i>';
            predictBtn.disabled = false;
        }
    });

    resetBtn.addEventListener('click', () => {
        resultContainer.classList.add('hidden');
        resultContainer.classList.remove('fade-in');
        form.reset();
    });

    function showResult(data) {
        resultContainer.classList.remove('hidden');
        // Trigger reflow
        void resultContainer.offsetWidth;
        resultContainer.classList.add('fade-in');

        const probability = (data.probability * 100).toFixed(1);
        const adviceText = document.getElementById('adviceText');

        // Update Text Content
        resultTitle.textContent = data.result; // Tier name from backend
        probabilityText.textContent = `${probability}% Risk Probability`;

        // Ensure advice element exists (if not added to HTML yet, we might need to add it dynamically or update HTML)
        if (adviceText) {
            adviceText.textContent = data.advice;
        } else {
            // Create if missing
            const p = document.createElement('p');
            p.id = 'adviceText';
            p.style.marginTop = '15px';
            p.style.color = '#57606f';
            p.textContent = data.advice;
            document.querySelector('.probability-meter').appendChild(p);
        }

        // Apply Styles based on class from backend
        // Reset classes first
        resultTitle.className = '';
        resultTitle.classList.add(data.tier_class);

        // Define colors/icons map for frontend or rely on backend class mapping
        // Let's use simple logic based on probability for colors here for immediate feedback
        // Or better, use specific styles for the classes we defined in CSS

        let color, icon;
        if (data.probability < 0.20) {
            color = "#2ecc71"; // Green
            icon = '<i class="fa-solid fa-seedling"></i>';
        } else if (data.probability < 0.45) {
            color = "#3498db"; // Blue
            icon = '<i class="fa-solid fa-shield-halved"></i>';
        } else if (data.probability < 0.65) {
            color = "#f1c40f"; // Yellow
            icon = '<i class="fa-solid fa-triangle-exclamation"></i>';
        } else if (data.probability < 0.85) {
            color = "#e67e22"; // Orange
            icon = '<i class="fa-solid fa-fire"></i>';
        } else {
            color = "#e74c3c"; // Red
            icon = '<i class="fa-solid fa-skull-crossbones"></i>';
        }

        resultTitle.style.color = color;
        resultIcon.innerHTML = icon;
        resultIcon.style.color = color;
        meterFill.style.backgroundColor = color;

        // Render Risk Factors
        const riskFactorsContainer = document.getElementById('riskFactors');
        riskFactorsContainer.innerHTML = '<h3>Key Drivers</h3>';

        if (data.impact_factors && data.impact_factors.length > 0) {
            data.impact_factors.forEach(factor => {
                const item = document.createElement('div');
                item.className = 'impact-item';

                const directionColor = factor.sign > 0 ? '#e74c3c' : '#2ecc71';
                const directionIcon = factor.sign > 0 ? '<i class="fa-solid fa-arrow-trend-up"></i>' : '<i class="fa-solid fa-arrow-trend-down"></i>';

                item.innerHTML = `
                    <div class="impact-header">
                        <span class="impact-name">${factor.factor}</span>
                        <span class="impact-direction" style="color: ${directionColor}">
                            ${directionIcon} ${factor.direction}
                        </span>
                    </div>
                    <div class="impact-bar-bg">
                        <div class="impact-bar" style="width: ${factor.strength}%; background-color: ${directionColor}"></div>
                    </div>
                `;
                riskFactorsContainer.appendChild(item);
            });
        } else {
            riskFactorsContainer.innerHTML += '<p class="impact-none">No significant specific drivers found.</p>';
        }

        // Animate meter
        setTimeout(() => {
            meterFill.style.width = `${probability}%`;
            // Animate bars
            document.querySelectorAll('.impact-bar').forEach(bar => {
                const width = bar.style.width;
                bar.style.width = '0';
                setTimeout(() => bar.style.width = width, 100);
            });
        }, 300);
    }
});
