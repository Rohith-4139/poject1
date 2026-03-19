// ====================== COMMON EFFECTS ======================

// Create particles (only if container exists on this page)
const particlesContainer = document.getElementById('particles');
if (particlesContainer) {
  for (let i = 0; i < 40; i++) {
    const p = document.createElement('div');
    p.className = 'particle';
    p.style.left = Math.random() * 100 + '%';
    p.style.animationDelay = Math.random() * 20 + 's';
    particlesContainer.appendChild(p);
  }
}

// ====================== LOGIN PAGE LOGIC ======================

// Authorized credentials (demo)
const AUTHORIZED_EMAIL = 'user@hypertension.com';
const AUTHORIZED_PASSWORD = 'password123';

// Toggle password visibility
function togglePasswordVisibility() {
  const passwordInput = document.getElementById('password');
  const toggleIcon = document.getElementById('togglePassword');

  if (!passwordInput || !toggleIcon) return;

  if (passwordInput.type === 'password') {
    passwordInput.type = 'text';
    toggleIcon.classList.remove('bx-hide');
    toggleIcon.classList.add('bx-show');
  } else {
    passwordInput.type = 'password';
    toggleIcon.classList.remove('bx-show');
    toggleIcon.classList.add('bx-hide');
  }
}

// Show error message
function showError(message) {
  const errorDiv = document.getElementById('errorMessage');
  if (!errorDiv) return;

  errorDiv.textContent = message;
  errorDiv.style.display = 'block';
  setTimeout(() => {
    errorDiv.style.display = 'none';
  }, 5000);
}

// Show success and redirect to language page
function showSuccessAndRedirect() {
  const modal = document.getElementById('successModal');
  if (modal) {
    modal.classList.add('show');
  }

  setTimeout(() => {
    // after login/signup/guest -> go to language selection
    window.location.href = '/language';
  }, 2000);
}

// Handle login
function handleLogin() {
  const emailInput = document.getElementById('email');
  const passwordInput = document.getElementById('password');
  const loginBtn = document.getElementById('loginBtn');

  if (!emailInput || !passwordInput || !loginBtn) return; // not on login page

  const email = emailInput.value.trim();
  const password = passwordInput.value;

  if (!email || !password) {
    showError('Please fill in all fields');
    return;
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    showError('Please enter a valid email address');
    return;
  }

  loginBtn.textContent = 'Signing in...';
  loginBtn.disabled = true;

  setTimeout(() => {
    if (email === AUTHORIZED_EMAIL && password === AUTHORIZED_PASSWORD) {
      sessionStorage.setItem('userEmail', email);
      sessionStorage.setItem('isLoggedIn', 'true');
      showSuccessAndRedirect();
    } else {
      showError(
        'Invalid email or password. Try: user@hypertension.com / password123'
      );
      loginBtn.textContent = 'Sign in';
      loginBtn.disabled = false;
    }
  }, 1500);
}

// Handle Google login (demo only)
function handleGoogleLogin() {
  const googleBtn = document.querySelector('.google-btn');
  if (!googleBtn) return; // not on login page

  googleBtn.style.opacity = '0.6';
  googleBtn.style.pointerEvents = 'none';

  setTimeout(() => {
    sessionStorage.setItem('userEmail', 'google.user@gmail.com');
    sessionStorage.setItem('isLoggedIn', 'true');
    sessionStorage.setItem('loginMethod', 'google');
    showSuccessAndRedirect();
  }, 1500);
}

// Handle Forgot Password
function handleForgotPassword(event) {
  event.preventDefault();
  window.location.href = '/forgot-password';
}

// Enter key to submit (only if fields exist)
const passwordField = document.getElementById('password');
if (passwordField) {
  passwordField.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') handleLogin();
  });
}

const emailField = document.getElementById('email');
if (emailField) {
  emailField.addEventListener('keypress', function (e) {
    if (e.key === 'Enter') handleLogin();
  });
}

// ====================== LANGUAGE PAGE LOGIC ======================

function confirmLanguage(event) {
  const select = document.getElementById('languageSelect');
  if (!select) return; // not on language page

  const selectedLanguage = select.value;            // en, hi, te, ta, bn
  const languageName = select.selectedOptions[0].text; // English, Hindi, ...

  // store choice so React can read from sessionStorage
  sessionStorage.setItem('selectedLanguage', selectedLanguage);
  sessionStorage.setItem('languageName', languageName);

  const btn = event.target;
  btn.innerHTML = '<i class="bx bx-loader-alt bx-spin"></i> Loading dashboard...';
  btn.disabled = true;

  // after selecting language, go directly to React dashboard
  setTimeout(() => {
    window.location.href = 'http://localhost:3000/'; // React dev server URL
  }, 800);
}

// ====================== SIGNUP MODAL ======================

function openSignupModal(event) {
  if (event) event.preventDefault();
  const modal = document.getElementById('signupModal');
  if (modal) modal.classList.add('show');
}

function closeSignupModal() {
  const modal = document.getElementById('signupModal');
  if (modal) modal.classList.remove('show');
}

function handleSignup() {
  const emailInput = document.getElementById('signupEmail');
  const pass1Input = document.getElementById('signupPassword');
  const pass2Input = document.getElementById('signupPassword2');
  const errorBox = document.getElementById('signupError');
  const btn = document.getElementById('signupBtn');

  if (!emailInput || !pass1Input || !pass2Input || !errorBox || !btn) return;

  const email = emailInput.value.trim();
  const pass1 = pass1Input.value;
  const pass2 = pass2Input.value;

  if (!email || !pass1 || !pass2) {
    errorBox.textContent = 'Please fill all the fields.';
    errorBox.style.display = 'block';
    return;
  }

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(email)) {
    errorBox.textContent = 'Please enter a valid email.';
    errorBox.style.display = 'block';
    return;
  }

  if (pass1 !== pass2) {
    errorBox.textContent = 'Passwords do not match.';
    errorBox.style.display = 'block';
    return;
  }

  errorBox.style.display = 'none';
  btn.textContent = 'Creating...';
  btn.disabled = true;

  // fake signup – here you can call backend later
  setTimeout(() => {
    sessionStorage.setItem('userEmail', email);
    sessionStorage.setItem('isLoggedIn', 'true');
    closeSignupModal();
    showSuccessAndRedirect();
    btn.textContent = 'Create Account';
    btn.disabled = false;
  }, 1500);
}

// ====================== GUEST CONFIRM MODAL ======================

function openGuestConfirm() {
  const modal = document.getElementById('guestModal');
  if (modal) modal.classList.add('show');
}

function closeGuestModal() {
  const modal = document.getElementById('guestModal');
  if (modal) modal.classList.remove('show');
}

function confirmGuest() {
  sessionStorage.setItem('isGuest', 'true');
  sessionStorage.setItem('isLoggedIn', 'true');
  closeGuestModal();
  showSuccessAndRedirect();
}
