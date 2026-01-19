const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// Cache for genre queries (client-side)
const genreCache = {};

function card(node, movie) {
  const tpl = document.getElementById('card-template');
  const el = tpl.content.firstElementChild.cloneNode(true);
  if (movie.movieId) el.dataset.mid = String(movie.movieId);
  el.querySelector('.title').textContent = movie.title ?? movie.movieId;
  const sub = [];
  if (movie.genres) sub.push(movie.genres);
  if (movie.year) sub.push(String(movie.year));
  if (movie.score != null) sub.push(`★ ${movie.score.toFixed(2)}`);
  el.querySelector('.sub').textContent = sub.join(' • ');
  const badge = el.querySelector('.badge');
  if (movie.score != null) {
    badge.textContent = `★ ${Number(movie.score).toFixed(2)}`;
    badge.hidden = false;
  }
  if (movie.poster) {
    const p = el.querySelector('.poster');
    p.classList.add('has-img');
    p.style.backgroundImage = `url('${movie.poster}')`;
    p.textContent = '';
  }
  el.addEventListener('mouseenter', (ev) => showTooltip(ev, movie));
  el.addEventListener('mousemove', (ev) => showTooltip(ev, movie));
  el.addEventListener('mouseleave', hideTooltip);
  // Click feedback disabled to keep focus on list (wishlist) only
  // favorites toggle
  const favBtn = el.querySelector('.fav');
  const favs = getFavorites();
  if (favs.has(movie.movieId)) favBtn.classList.add('active'), favBtn.textContent = '❤';
  favBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    const set = getFavorites();
    if (set.has(movie.movieId)) {
      set.delete(movie.movieId);
      favBtn.classList.remove('active');
      favBtn.textContent = '♡';
    } else {
      set.add(movie.movieId);
      favBtn.classList.add('active');
      favBtn.textContent = '❤';
      sendFeedback('list', movie.movieId);
    }
    saveFavorites(set);
    renderMyList();
  });
  node.appendChild(el);
}

async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return await r.json();
}

async function browseByGenre(genre) {
  // Fast genre browse with client-side caching
  const row = $('#browse-row');
  
  // Check cache first
  if (genreCache[genre]) {
    row.innerHTML = '';
    genreCache[genre].forEach((m) => card(row, m));
    const posters = await postersFor(genreCache[genre]);
    genreCache[genre].forEach(m => {
      const p = row.querySelector(`[data-mid="${CSS.escape(String(m.movieId))}"] .poster`);
      if (p && posters[m.movieId]) { p.classList.add('has-img'); p.style.backgroundImage = `url('${posters[m.movieId]}')`; p.textContent=''; }
    });
    return;
  }
  
  showSkeleton(row, 8);
  try {
    const data = await fetchJSON(`/movies?topN=50&genres=${encodeURIComponent(genre)}`);
    genreCache[genre] = data;  // Cache result
    row.innerHTML = '';
    data.forEach((m) => card(row, m));
    const posters = await postersFor(data);
    data.forEach(m => {
      const p = row.querySelector(`[data-mid="${CSS.escape(String(m.movieId))}"] .poster`);
      if (p && posters[m.movieId]) { p.classList.add('has-img'); p.style.backgroundImage = `url('${posters[m.movieId]}')`; p.textContent=''; }
    });
  } catch (e) {
    row.textContent = `Failed to load ${genre} movies`;
  }
}

async function loadPopular() {
  const row = $('#popular-row');
  showSkeleton(row, 8);
  try {
    const data = await fetchJSON('/popular?topN=50');
    row.innerHTML = '';
    data.forEach((m) => card(row, m));
    const posters = await postersFor(data);
    data.forEach(m => {
      const p = row.querySelector(`[data-mid="${CSS.escape(String(m.movieId))}"] .poster`);
      if (p && posters[m.movieId]) { p.classList.add('has-img'); p.style.backgroundImage = `url('${posters[m.movieId]}')`; p.textContent=''; }
    });
  } catch (e) {
    row.textContent = 'Failed to load popular titles';
  }
}

async function loadGenres() {
  const list = $('#genres-list');
  list.innerHTML = '';
  const rows = $('#genre-rows');
  rows.innerHTML = '';
  try {
    const gsrc = await fetchJSON('/genres');
    let genres = Array.isArray(gsrc) ? gsrc.slice() : [];
    // Deduplicate (case-insensitive)
    const seen = new Set();
    genres = genres.filter(g => { const k = g.toLowerCase(); if (seen.has(k)) return false; seen.add(k); return true; });
    // Sort alphabetically but push any variant of "no genre listed" to the end
    const isNone = (g) => g.toLowerCase().includes('no genre');
    genres.sort((a,b) => {
      const an = isNone(a), bn = isNone(b);
      if (an && !bn) return 1;
      if (!an && bn) return -1;
      return a.localeCompare(b);
    });
    const top = genres.slice(0, 10);
    top.forEach((g, i) => {
      const b = document.createElement('button');
      b.className = 'chip' + (i === 0 ? ' active' : '');
      b.textContent = g;
      b.onclick = () => { window.location.href = `/ui/genre.html?g=${encodeURIComponent(g)}`; };
      list.appendChild(b);
    });
    // Do not auto-render a horizontal row; use dedicated genre pages
    // Populate sidebar genres
    const side = document.getElementById('side-genres');
    if (side) {
      side.innerHTML = '';
      genres.forEach((g) => {
        const a = document.createElement('a');
        a.href = `/ui/genre.html?g=${encodeURIComponent(g)}`;
        a.className = 'side-link';
        a.textContent = g;
        side.appendChild(a);
      });
      const title = document.getElementById('side-genres-title');
      if (title) {
        title.style.cursor = 'pointer';
        title.onclick = () => {
          side.classList.toggle('collapsed');
          title.textContent = side.classList.contains('collapsed') ? 'Genres ▸' : 'Genres ▾';
        };
      }
    }
  } catch (e) {
    list.textContent = 'No genres available';
  }
}

function currentUserId() {
  const uid = (localStorage.getItem('ml_user') || 'guest').trim();
  return uid || 'guest';
}

function updateGreeting() {
  const greet = document.getElementById('user-greeting');
  if (!greet) return;
  greet.textContent = `Hi, ${currentUserId()}`;
}

// Set greeting immediately on script load to avoid a guest flash
updateGreeting();

function currentPosterSize() {
  const val = localStorage.getItem('ml_poster_quality');
  return val || 'w342';
}

function applyItemLayout() {
  const stored = localStorage.getItem('ml_items_per_row');
  const n = Math.max(3, Math.min(10, Number(stored) || 6));
  document.documentElement.style.setProperty('--items-per-row', n);
}

function applyTheme() {
  const t = (localStorage.getItem('ml_theme') || 'dark').trim();
  const theme = t === 'light' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', theme);
}

async function renderGenre(genre) {
  const rows = $('#genre-rows');
  rows.innerHTML = '';
  const section = document.createElement('div');
  const h = document.createElement('h3');
  h.textContent = genre;
  section.appendChild(h);
  const row = document.createElement('div');
  row.className = 'row';
  section.appendChild(row);
  rows.appendChild(section);
  try {
    const data = await fetchJSON(`/popular?topN=20&genres=${encodeURIComponent(genre)}`);
    data.forEach((m) => card(row, m));
    const posters = await postersFor(data);
    data.forEach(m => {
      const p = row.querySelector(`[data-mid="${CSS.escape(String(m.movieId))}"] .poster`);
      if (p && posters[m.movieId]) { p.classList.add('has-img'); p.style.backgroundImage = `url('${posters[m.movieId]}')`; p.textContent=''; }
    });
  } catch (e) {
    row.textContent = 'Failed to load';
  }
}

async function handleUserForm() {
  const form = $('#user-form');
  const recommendBtn = document.getElementById('recommend-btn');
  if (recommendBtn) {
    recommendBtn.addEventListener('click', () => form.requestSubmit());
  }
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const userId = (localStorage.getItem('ml_user') || '').trim();
    const topN = 20;
    const genres = $('#genres').value.trim();
    const yearFromEl = document.getElementById('yearFrom');
    const yearToEl = document.getElementById('yearTo');
    const yearFrom = yearFromEl ? String((yearFromEl.value||'').trim()) : '';
    const yearTo = yearToEl ? String((yearToEl.value||'').trim()) : '';
    const row = $('#user-row');
    localStorage.setItem('ml_user', userId || '');
    showSkeleton(row, 8);
    const params = new URLSearchParams({ topN: String(topN) });
    if (genres) params.set('genres', genres);
    if (yearFrom) params.set('year_from', yearFrom);
    if (yearTo) params.set('year_to', yearTo);
    try {
      const data = userId
        ? await fetchJSON(`/recommendations/user/${encodeURIComponent(userId)}?${params}`)
        : await fetchJSON(`/popular?${params}`);
      if (!data.length) {
        row.textContent = 'No recommendations found for this user.';
      } else {
        row.innerHTML = '';
        data.forEach((m) => card(row, m));
        const posters = await postersFor(data);
        data.forEach(m => {
          const p = row.querySelector(`[data-mid="${CSS.escape(String(m.movieId))}"] .poster`);
          if (p && posters[m.movieId]) { p.classList.add('has-img'); p.style.backgroundImage = `url('${posters[m.movieId]}')`; p.textContent=''; }
        });
      }
      // no continue-watching UI
    } catch (e) {
      row.textContent = 'Failed to load recommendations';
    }
  });
}

async function postersFor(items) {
  try {
    const ids = items.slice(0, 20).map((m) => m.movieId).filter(Boolean);
    if (!ids.length) return {};
    const size = currentPosterSize();
    const params = new URLSearchParams({ movieIds: ids.join(','), size });
    return await fetchJSON(`/posters?${params.toString()}`);
  } catch {
    return {};
  }
}

// Dual-thumb year slider (custom) using a single track
async function initYearSelectors() {
  const hiddenFrom = document.getElementById('yearFrom');
  const hiddenTo = document.getElementById('yearTo');
  const slider = document.getElementById('year-slider');
  const fill = document.getElementById('year-range-fill');
  const thumbFrom = document.getElementById('year-thumb-from');
  const thumbTo = document.getElementById('year-thumb-to');
  const label = document.getElementById('year-range-label');
  const labelFrom = document.getElementById('year-range-from');
  const labelTo = document.getElementById('year-range-to');
  if (!hiddenFrom || !hiddenTo || !slider || !fill || !thumbFrom || !thumbTo || !label || !labelFrom || !labelTo) return;

  let minYear = 1900;
  let maxYear = new Date().getFullYear();
  try {
    const data = await fetchJSON('/years');
    if (Number.isFinite(data.min)) minYear = data.min;
    if (Number.isFinite(data.max)) maxYear = data.max;
  } catch {
    // fall back to defaults
  }

  let fromVal = minYear;
  let toVal = maxYear;

  const clamp = (v) => Math.min(maxYear, Math.max(minYear, Math.round(v)));
  const pct = (v) => ((v - minYear) / (maxYear - minYear)) * 100;

  function render() {
    const isFullRange = fromVal === minYear && toVal === maxYear;
    hiddenFrom.value = isFullRange ? '' : String(fromVal);
    hiddenTo.value = isFullRange ? '' : String(toVal);
    label.textContent = isFullRange ? 'All years' : 'Custom range';
    labelFrom.textContent = String(fromVal);
    labelTo.textContent = String(toVal);
    const fromPct = pct(fromVal);
    const toPct = pct(toVal);
    fill.style.left = `${fromPct}%`;
    fill.style.width = `${toPct - fromPct}%`;
    thumbFrom.style.left = `${fromPct}%`;
    thumbTo.style.left = `${toPct}%`;
  }

  function setFrom(newVal) {
    fromVal = clamp(newVal);
    if (fromVal > toVal) toVal = fromVal;
    render();
  }
  function setTo(newVal) {
    toVal = clamp(newVal);
    if (toVal < fromVal) fromVal = toVal;
    render();
  }

  function clientXToValue(clientX) {
    const rect = slider.getBoundingClientRect();
    const ratio = (clientX - rect.left) / rect.width;
    return clamp(minYear + ratio * (maxYear - minYear));
  }

  let active = null;
  function onPointerMove(ev) {
    if (!active) return;
    ev.preventDefault();
    const val = clientXToValue(ev.clientX);
    if (active === 'from') setFrom(val); else setTo(val);
  }
  function stopPointer() {
    active = null;
    window.removeEventListener('pointermove', onPointerMove);
    window.removeEventListener('pointerup', stopPointer);
  }
  function startPointer(which, ev) {
    active = which;
    onPointerMove(ev);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', stopPointer);
  }

  thumbFrom.addEventListener('pointerdown', (ev) => { ev.preventDefault(); thumbFrom.setPointerCapture(ev.pointerId); startPointer('from', ev); });
  thumbTo.addEventListener('pointerdown', (ev) => { ev.preventDefault(); thumbTo.setPointerCapture(ev.pointerId); startPointer('to', ev); });

  // Keyboard support
  function handleKey(ev, which) {
    const delta = (ev.shiftKey ? 10 : 1) * (ev.key === 'ArrowRight' || ev.key === 'ArrowUp' ? 1 : ev.key === 'ArrowLeft' || ev.key === 'ArrowDown' ? -1 : 0);
    if (delta === 0) return;
    ev.preventDefault();
    if (which === 'from') setFrom(fromVal + delta); else setTo(toVal + delta);
  }
  thumbFrom.addEventListener('keydown', (ev) => handleKey(ev, 'from'));
  thumbTo.addEventListener('keydown', (ev) => handleKey(ev, 'to'));

  // Click on track moves closest thumb
  slider.addEventListener('pointerdown', (ev) => {
    const val = clientXToValue(ev.clientX);
    const distFrom = Math.abs(val - fromVal);
    const distTo = Math.abs(val - toVal);
    if (distFrom <= distTo) setFrom(val); else setTo(val);
  });

  render();
}

// Build genre chips and year preset chips under the Recommendations panel
async function initUserFilters() {
  // Genre chips populate the hidden text input #genres for compatibility
  const box = document.getElementById('user-genres');
  if (box) {
    try {
      const gsrc = await fetchJSON('/genres');
      let genres = Array.isArray(gsrc) ? gsrc.slice() : [];
      const seen2 = new Set();
      genres = genres.filter(g => { const k = g.toLowerCase(); if (seen2.has(k)) return false; seen2.add(k); return true; });
      const isNone = (g) => g.toLowerCase().includes('no genre');
      genres.sort((a, b) => {
        const an = isNone(a), bn = isNone(b);
        if (an && !bn) return 1;
        if (!an && bn) return -1;
        return a.localeCompare(b);
      });
      const idx = genres.findIndex(isNone);
      if (idx >= 0) { const [t] = genres.splice(idx, 1); genres.push(t); }
      const picked = new Set();
      genres.slice(0, 20).forEach((g) => {
        const b = document.createElement('button');
        b.type = 'button';
        b.className = 'chip';
        b.textContent = g;
        b.onclick = () => {
          if (picked.has(g)) { picked.delete(g); b.classList.remove('active'); }
          else { picked.add(g); b.classList.add('active'); }
          const inp = document.getElementById('genres');
          if (inp) inp.value = Array.from(picked).join(', ');
        };
        box.appendChild(b);
      });
    } catch {
      // ignore
    }
  }

  // Year presets removed in favor of dual range sliders
}

function handleBrowse() {
  const form = $('#browse-form');
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const q = $('#q').value.trim();
    const topN = 50;
    const row = $('#browse-row');
    showSkeleton(row, 8);
    try {
      const data = await fetchJSON(`/movies?topN=${topN}${q ? `&q=${encodeURIComponent(q)}` : ''}`);
      row.innerHTML = '';
      if (!data.length) { row.textContent = 'No results.'; return; }
      data.forEach((m) => card(row, m));
      const posters = await postersFor(data);
      data.forEach(m => {
        const p = row.querySelector(`[data-mid="${CSS.escape(String(m.movieId))}"] .poster`);
        if (p && posters[m.movieId]) { p.classList.add('has-img'); p.style.backgroundImage = `url('${posters[m.movieId]}')`; p.textContent=''; }
      });
    } catch (e) {
      row.textContent = 'Search failed';
    }
  });
  // Enter to search
  $('#q').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); form.requestSubmit(); } });
}

window.addEventListener('DOMContentLoaded', async () => {
  updateGreeting();
  applyTheme();
  applyItemLayout();
  // Load genres FIRST before other UI
  await loadGenres();
  handleUserForm();
  initUserFilters();
  initYearSelectors();
  handleBrowse();
  await loadPopular();
  await loadMostClicked();
  renderMyList();
  enableKeyboardScroll();
  // TopN control
  initTopN();
});

// Favorites (localStorage)
const FAV_KEY_PREFIX = 'ml_favorites_';
function favoritesKey() { return `${FAV_KEY_PREFIX}${currentUserId()}`; }
function getFavorites() {
  try { return new Set(JSON.parse(localStorage.getItem(favoritesKey()) || '[]')); } catch { return new Set(); }
}
function saveFavorites(set) {
  localStorage.setItem(favoritesKey(), JSON.stringify(Array.from(set)));
}
function refreshFavoriteButtons() {
  const favs = getFavorites();
  $$('.card').forEach((cardEl) => {
    const mid = cardEl.dataset.mid;
    if (!mid) return;
    const btn = cardEl.querySelector('.fav');
    if (!btn) return;
    const active = favs.has(Number(mid)) || favs.has(mid);
    btn.classList.toggle('active', active);
    btn.textContent = active ? '❤' : '♡';
  });
}
async function renderMyList() {
  const row = document.getElementById('mylist-row');
  if (!row) return;
  row.innerHTML = '';
  const ids = Array.from(getFavorites());
  if (!ids.length) { const d = document.createElement('div'); d.className='empty-center'; d.textContent='Add titles with the heart to build your list.'; row.appendChild(d); return; }
  try {
    const items = await fetchJSON(`/movies/by_ids?movieIds=${encodeURIComponent(ids.join(','))}`);
    const posters = await postersFor(items);
    items.map((m) => ({...m, poster: posters[m.movieId]})).forEach((m) => card(row, m));
  } catch { row.textContent = 'Failed to load list'; }
}

// Keyboard row scrolling
function enableKeyboardScroll() {
  const rows = $$('.row');
  rows.forEach((r) => r.addEventListener('mouseenter', () => rows.forEach((x) => x.classList.toggle('active', x===r))));
  window.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    const active = document.querySelector('.row.active');
    if (!active) return;
    const dx = e.key === 'ArrowRight' ? 300 : -300;
    active.scrollBy({ left: dx, behavior: 'smooth' });
  });
}

// TopN chips + arrows
function initTopN() {
  const chipsBox = document.getElementById('topn-chips');
  const left = document.getElementById('topn-left');
  const right = document.getElementById('topn-right');
  const hidden = document.getElementById('topN');
  if (!chipsBox || !left || !right || !hidden) return;
  const values = [5,10,15,20];
  let idx = Math.max(0, values.indexOf(Number(hidden.value) || 20));
  function render() {
    chipsBox.innerHTML='';
    values.forEach((v,i)=>{
      const b=document.createElement('button'); b.type='button'; b.className='chip'+(i===idx?' active':''); b.textContent=String(v);
      b.onclick=()=>{ idx=i; hidden.value=String(values[idx]); render(); };
      chipsBox.appendChild(b);
    });
  }
  left.onclick = () => { idx = Math.max(0, idx-1); hidden.value=String(values[idx]); render(); };
  right.onclick = () => { idx = Math.min(values.length-1, idx+1); hidden.value=String(values[idx]); render(); };
  render();
}

// Continue Watching
async function renderContinue(userId) {
  const row = document.getElementById('continue-row');
  if (!row) return;
  showSkeleton(row, 6);
  try {
    const evs = await fetchJSON(`/history?userId=${encodeURIComponent(userId)}&topN=50`);
    const ids = Array.from(new Set(evs.map((e) => e.movieId))).slice(0, 20);
    if (!ids.length) { row.textContent = 'No recent activity.'; return; }
    const items = await fetchJSON(`/movies/by_ids?movieIds=${encodeURIComponent(ids.join(','))}`);
    const posters = await postersFor(items);
    row.innerHTML='';
    items.map((m) => ({...m, poster: posters[m.movieId]})).forEach((m) => card(row, m));
  } catch { row.textContent = 'No recent activity.'; }
}

// Skeleton helpers
function showSkeleton(row, n) {
  row.innerHTML = '';
  for (let i = 0; i < n; i++) {
    const d = document.createElement('div');
    d.className = 'card skeleton';
    const p = document.createElement('div');
    p.className = 'poster';
    const t = document.createElement('div');
    t.className = 'meta title';
    d.appendChild(p); d.appendChild(t);
    row.appendChild(d);
  }
}

// Most Clicked (global) and per-user
async function loadMostClicked() {
  const row = document.getElementById('clicked-row');
  if (!row) return;
  showSkeleton(row, 8);
  try {
    const items = await fetchJSON('/feedback/summary?topN=20&window_days=30&half_life_days=14&w_click=1&w_list=3&blend_baseline=0.2');
    const posters = await postersFor(items);
    row.innerHTML='';
    items.map((m) => ({...m, poster: posters[m.movieId]})).forEach((m) => card(row, m));
  } catch { row.textContent = 'No click data yet.'; }
}

async function loadMostClickedForUser(userId) {
  const cont = document.getElementById('continue-row'); // ensure continued loaded separately
  try {
    const row = document.getElementById('user-clicked-row');
    if (!row) return;
    showSkeleton(row, 8);
    const items = await fetchJSON(`/feedback/summary?userId=${encodeURIComponent(userId)}&topN=20`);
    const posters = await postersFor(items);
    row.innerHTML='';
    items.map((m) => ({...m, poster: posters[m.movieId]})).forEach((m) => card(row, m));
  } catch {}
}

// Tooltip
const tooltip = document.getElementById('tooltip');
function showTooltip(ev, movie) {
  if (!tooltip) return;
  $('.tt-title', tooltip).textContent = movie.title ?? movie.movieId;
  const bits = [];
  if (movie.year) bits.push(movie.year);
  if (movie.genres) bits.push(movie.genres);
  if (movie.score != null) bits.push(`score ${Number(movie.score).toFixed(2)}`);
  $('.tt-sub', tooltip).textContent = bits.join(' • ');
  tooltip.hidden = false;
  const pad = 12;
  const x = Math.min(window.innerWidth - tooltip.offsetWidth - pad, ev.clientX + pad);
  const y = Math.min(window.innerHeight - tooltip.offsetHeight - pad, ev.clientY + pad);
  tooltip.style.left = x + 'px';
  tooltip.style.top = y + 'px';
}
function hideTooltip() { if (tooltip) tooltip.hidden = true; }

// Feedback helper
async function sendFeedback(action, movieId) {
  try {
    const uid = currentUserId();
    await fetch('/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ userId: uid, movieId, action }) });
  } catch {}
}

// Modal functionality
function openModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) modal.hidden = false;
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) modal.hidden = true;
}

// Settings Modal
const settingsBtn = document.getElementById('settings-btn');
const saveSettingsBtn = document.getElementById('save-settings');
if (settingsBtn) {
  settingsBtn.addEventListener('click', () => {
    openModal('settings-modal');
    // Load saved settings
    const posterQuality = localStorage.getItem('ml_poster_quality') || 'w342';
    const itemsPerRow = localStorage.getItem('ml_items_per_row') || '6';
    const autoPlay = localStorage.getItem('ml_auto_play') === 'true';
    const theme = localStorage.getItem('ml_theme') || 'dark';
    
    document.getElementById('poster-quality').value = posterQuality;
    document.getElementById('items-per-row').value = itemsPerRow;
    document.getElementById('auto-play-trailers').checked = autoPlay;
    const themeSelect = document.getElementById('theme-select');
    if (themeSelect) themeSelect.value = theme;
  });
}

if (saveSettingsBtn) {
  saveSettingsBtn.addEventListener('click', () => {
    const posterQuality = document.getElementById('poster-quality').value;
    const itemsPerRow = document.getElementById('items-per-row').value;
    const autoPlay = document.getElementById('auto-play-trailers').checked;
    const themeSelect = document.getElementById('theme-select');
    const theme = themeSelect ? themeSelect.value : 'dark';
    
    localStorage.setItem('ml_poster_quality', posterQuality);
    localStorage.setItem('ml_items_per_row', itemsPerRow);
    localStorage.setItem('ml_auto_play', autoPlay);
    localStorage.setItem('ml_theme', theme);
    
    // Apply items per row setting
    applyItemLayout();
    applyTheme();
    
    closeModal('settings-modal');
    console.log('Settings saved');
  });
}

// Login Modal
const authBtn = document.getElementById('auth-btn');
const loginBtn = document.getElementById('login-btn');
const logoutBtn = document.getElementById('logout-btn');

function hideAllModals() {
  $$('.modal').forEach(m => m.hidden = true);
}

if (authBtn) {
  authBtn.addEventListener('click', () => {
    hideAllModals();
    openModal('login-modal');
    const currentUser = localStorage.getItem('ml_user') || 'guest';
    document.getElementById('current-user').textContent = currentUser;
    document.getElementById('user-id-input').value = currentUser === 'guest' ? '' : currentUser;
    updateGreeting();
  });
}

// Ensure all modals are hidden on load
document.addEventListener('DOMContentLoaded', hideAllModals);

if (loginBtn) {
  loginBtn.addEventListener('click', () => {
    const userId = document.getElementById('user-id-input').value.trim();
    if (userId) {
      localStorage.setItem('ml_user', userId);
      document.getElementById('current-user').textContent = userId;
      closeModal('login-modal');
      updateGreeting();
      renderMyList();
      refreshFavoriteButtons();
    } else {
      alert('Please enter a valid user ID');
    }
  });
}

if (logoutBtn) {
  logoutBtn.addEventListener('click', () => {
    localStorage.setItem('ml_user', 'guest');
    document.getElementById('current-user').textContent = 'guest';
    document.getElementById('user-id-input').value = '';
    closeModal('login-modal');
    updateGreeting();
    renderMyList();
    refreshFavoriteButtons();
  });
}

// Close modal when clicking close button
$$('.modal-close').forEach(btn => {
  btn.addEventListener('click', (e) => {
    e.preventDefault();
    e.stopPropagation();
    const modalId = e.currentTarget.dataset.modal;
    if (modalId) closeModal(modalId);
  });
});

// Close modal when clicking outside
// Prevent clicks on modal content from closing the modal
$$('.modal-content').forEach(content => {
  content.addEventListener('click', (e) => {
    e.stopPropagation();
  });
});

// Close modal when clicking outside
$$('.modal').forEach(modal => {
  modal.addEventListener('click', (e) => {
    if (e.target === modal || e.target.classList.contains('modal')) {
      modal.hidden = true;
    }
  });
});

// ESC key to close modals
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    $$('.modal').forEach(m => m.hidden = true);
  }
});

// Analytics button
const analyticsBtn = document.getElementById('analytics-btn');
if (analyticsBtn) {
  analyticsBtn.addEventListener('click', () => {
    window.location.href = '/ui/analytics.html';
  });
}
