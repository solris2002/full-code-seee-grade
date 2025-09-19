document.addEventListener("DOMContentLoaded", () => {
  const track = document.querySelector(".slider-track");
  const slides = document.querySelectorAll(".slider-track .slide");
  const prevBtn = document.querySelector(".slider-btn.prev");
  const nextBtn = document.querySelector(".slider-btn.next");

  let currentIndex = 0;

  function updateSlider() {
    track.style.transform = `translateX(-${currentIndex * 100}%)`;
  }

  prevBtn.addEventListener("click", () => {
    currentIndex = (currentIndex - 1 + slides.length) % slides.length;
    updateSlider();
  });

  nextBtn.addEventListener("click", () => {
    currentIndex = (currentIndex + 1) % slides.length;
    updateSlider();
  });

  // Zoom
  const modal = document.getElementById("imageModal");
  const modalImg = document.getElementById("modalImg");
  const closeModal = document.querySelector(".modal .close");

  document.querySelectorAll(".zoom-btn").forEach(btn => {
    btn.addEventListener("click", (e) => {
      const img = e.target.closest(".slide").querySelector("img");
      modal.style.display = "block";
      modalImg.src = img.src;
    });
  });

  closeModal.addEventListener("click", () => {
    modal.style.display = "none";
  });

  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      modal.style.display = "none";
    }
  });
});

const btn = document.getElementById('theme-toggle');
const icon = btn.querySelector('.icon');

function applyTheme(theme) {
  document.body.classList.toggle('dark', theme === 'dark');
  localStorage.setItem('theme', theme);
  icon.textContent = theme === 'dark' ? '🌙' : '☀️';
  btn.setAttribute('aria-pressed', theme === 'dark' ? 'true' : 'false');
  btn.setAttribute('aria-label', theme === 'dark' ? 'Đang Dark mode. Bấm để chuyển Light mode' : 'Đang Light mode. Bấm để chuyển Dark mode');

  document.querySelectorAll('.slider .slide img').forEach(img => {
    const newSrc = theme === 'dark' ? img.dataset.dark : img.dataset.light;
    if (newSrc) img.src = newSrc;
  });
}

(function initTheme() {
  const saved = localStorage.getItem('theme');
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  applyTheme(saved || (prefersDark ? 'dark' : 'light'));
})();

btn.addEventListener('click', () => {
  const currentDark = document.body.classList.contains('dark');
  const next = currentDark ? 'light' : 'dark';
  btn.classList.add('spin');
  applyTheme(next);
  setTimeout(() => btn.classList.remove('spin'), 520);
});