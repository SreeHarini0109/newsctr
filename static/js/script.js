document.addEventListener('DOMContentLoaded', () => {

    // Initialize particles.js background
    particlesJS('particles-js', {
        particles: {
            number: { value: 60, density: { enable: true, value_area: 800 } },
            color: { value: '#ffffff' },
            shape: {
                type: 'circle',
                stroke: { width: 0, color: '#000000' },
            },
            opacity: {
                value: 0.3,
                random: true,
                anim: { enable: true, speed: 1, opacity_min: 0.1, sync: false }
            },
            size: {
                value: 3,
                random: true,
                anim: { enable: false, speed: 40, size_min: 0.1, sync: false }
            },
            line_linked: {
                enable: true,
                distance: 150,
                color: '#ffffff',
                opacity: 0.2,
                width: 1
            },
            move: {
                enable: true,
                speed: 1.5,
                direction: 'none',
                random: true,
                straight: false,
                out_mode: 'out',
                bounce: false,
                attract: { enable: false, rotateX: 600, rotateY: 1200 }
            }
        },
        interactivity: {
            detect_on: 'canvas',
            events: {
                onhover: { enable: true, mode: 'grab' },
                onclick: { enable: true, mode: 'push' },
                resize: true
            },
            modes: {
                grab: { distance: 140, line_linked: { opacity: 0.8 } },
                push: { particles_nb: 4 }
            }
        },
        retina_detect: true
    });

    // App Logic
    const btn = document.getElementById('predict-btn');
    const input = document.getElementById('headline-input');
    const resultsContainer = document.getElementById('results-container');
    const scoreCircle = document.getElementById('score-circle');
    const scorePercentage = document.getElementById('score-percentage');
    const engagementLabel = document.getElementById('engagement-label');
    const wordCount = document.getElementById('word-count');
    const charCount = document.getElementById('char-count');

    // UI State Helpers
    const setBtnState = (state) => {
        if (state === 'loading') {
            btn.innerHTML = '<span class="btn-text">ANALYZING...</span>';
            btn.disabled = true;
            btn.style.borderColor = 'var(--text-muted)';
            btn.style.color = 'var(--text-muted)';
        } else {
            btn.innerHTML = '<span class="btn-text">INITIALIZE PREDICTION</span>';
            btn.disabled = false;
            btn.style.borderColor = '';
            btn.style.color = '';
        }
    };

    const animateRing = (percentage) => {
        // Circumference of r=70 is approx 440
        const circumference = 440;
        const offset = circumference - (percentage / 100) * circumference;

        // Trigger reflow to restart anim if calculating multiple times
        scoreCircle.style.strokeDashoffset = '440';
        setTimeout(() => {
            scoreCircle.style.strokeDashoffset = offset;
        }, 50);

        // Counter animation
        let current = 0;
        const target = Math.round(percentage);
        const interval = setInterval(() => {
            if (current >= target) {
                clearInterval(interval);
                scorePercentage.textContent = target + '%';
            } else {
                current += 1;
                scorePercentage.textContent = current + '%';
            }
        }, 1500 / (target || 1)); // scale speed based on target

        // Update label and stroke color
        let color, labelText;
        if (percentage >= 75) {
            color = '#00ffcc'; // High - Neon Green
            labelText = 'VIRAL POTENTIAL';
        } else if (percentage >= 40) {
            color = 'var(--accent-blue)'; // Med - Neon Blue
            labelText = 'ENGAGING CONTENT';
        } else if (percentage >= 15) {
            color = '#ffcc00'; // Low Med - Yellow
            labelText = 'AVERAGE ATTRACTION';
        } else {
            color = '#ff3366'; // Low - Red
            labelText = 'LOW ENGAGEMENT';
        }

        scoreCircle.style.stroke = color;
        engagementLabel.style.color = color;
        engagementLabel.textContent = labelText;

        // Adjust CSS vars for glow
        document.documentElement.style.setProperty('--accent-glow', color + '66'); // add alpha
    };

    // Event Listener
    btn.addEventListener('click', async () => {
        const headline = input.value.trim();
        if (!headline) return;

        setBtnState('loading');

        // Hide results gracefully if predicting again
        resultsContainer.style.opacity = '0';

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ headline: headline })
            });

            const data = await response.json();

            if (data.error) throw new Error(data.error);

            // Update DOM
            wordCount.textContent = data.features.word_count;
            charCount.textContent = data.features.char_count;
            document.getElementById('model-value').textContent = data.features.model_used || "BERT + MLP";

            // Show results
            resultsContainer.classList.remove('hidden');
            setTimeout(() => {
                resultsContainer.style.opacity = '1';
                // Anim probability 0-100%
                animateRing(data.probability * 100);
            }, 300);

        } catch (err) {
            console.error(err);
            alert("Prediction failed: " + err.message);
        } finally {
            setBtnState('ready');
        }
    });

    // Enter key support
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            btn.click();
        }
    });
});
