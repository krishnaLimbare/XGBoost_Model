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
        data.family_history = parseInt(data.family_history);

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

        const isDiabetes = data.prediction === 1;
        const probability = (data.probability * 100).toFixed(1);

        if (isDiabetes) {
            resultTitle.textContent = "High Risk Detected";
            resultTitle.style.color = "#ff4757";
            resultIcon.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i>';
            resultIcon.style.color = "#ff4757";
            meterFill.style.backgroundColor = "#ff4757";
        } else {
            resultTitle.textContent = "Low Risk Detected";
            resultTitle.style.color = "#2ed573";
            resultIcon.innerHTML = '<i class="fa-solid fa-shield-heart"></i>';
            resultIcon.style.color = "#2ed573";
            meterFill.style.backgroundColor = "#2ed573";
        }

        // Animate meter
        setTimeout(() => {
            meterFill.style.width = `${probability}%`;
        }, 300);

        probabilityText.textContent = `${probability}% Risk Probability`;
    }
});
