const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

function card(node, movie) {
  const tpl = document.getElementById('card-template');
  const el = tpl.content.firstElementChild.cloneNode(true);
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

async function loadPopular() {
  const row = $('#popular-row');
  showSkeleton(row, 8);
  try {
    const data = await fetchJSON('/popular?topN=50');
    const posters = await postersFor(data);
    row.innerHTML = '';
    data.map((m) => ({...m, poster: posters[m.movieId]})).forEach((m) => card(row, m));
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
    genres = genres.filter(g => g.toLowerCase() !== 'no genre listed' && g.toLowerCase() !== '(genres not listed)');
    genres.push('(genres not listed)');
    const top = genres.slice(0, 10);
    top.forEach((g, i) => {
      const b = document.createElement('button');
      b.className = 'chip' + (i === 0 ? ' active' : '');
      b.textContent = g;
      b.onclick = () => {
        $$('.chip').forEach((c) => c.classList.remove('active'));
        b.classList.add('active');
        renderGenre(g);
      };
      list.appendChild(b);
    });
    if (top.length) renderGenre(top[0]);
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

async function initUsersSelect() {
  const headerSel = document.getElementById('userSelHeader');
  const formSel = document.getElementById('userSel');
  const targets = [headerSel, formSel].filter(Boolean);
  if (targets.length === 0) return;
  try {
    const users = await fetchJSON('/users?limit=500');
    targets.forEach(sel => {
      users.forEach(u => {
        const opt = document.createElement('option');
        opt.value = u; opt.textContent = u;
        sel.appendChild(opt);
      });
    });
    // keep header and (optional) form select in sync
    if (headerSel && formSel) {
      headerSel.addEventListener('change', () => { formSel.value = headerSel.value; });
      formSel.addEventListener('change', () => { headerSel.value = formSel.value; });
    }
  } catch {}
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
    const posters = await postersFor(data);
    data.map((m) => ({...m, poster: posters[m.movieId]})).forEach((m) => card(row, m));
  } catch (e) {
    row.textContent = 'Failed to load';
  }
}

async function handleUserForm() {
  const form = $('#user-form');
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const userSel = document.getElementById('userSelHeader');
    const userId = userSel ? String(userSel.value || '').trim() : '';
    const topN = Number($('#topN').value || 20);
    const genres = $('#genres').value.trim();
    const yearFrom = $('#yearFrom').value.trim();
    const yearTo = $('#yearTo').value.trim();
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
      const posters = await postersFor(data.slice(0, 12));
      const withPosters = data.map((m) => ({...m, poster: posters[m.movieId]}));
      if (!withPosters.length) {
        row.textContent = 'No recommendations found for this user.';
      } else {
        withPosters.forEach((m) => card(row, m));
      }
      // no continue-watching UI
    } catch (e) {
      row.textContent = 'Failed to load recommendations';
    }
  });
}

async function postersFor(items) {
  try {
    const ids = items.map((m) => m.movieId).filter(Boolean);
    if (!ids.length) return {};
    return await fetchJSON(`/posters?movieIds=${encodeURIComponent(ids.join(','))}`);
  } catch {
    return {};
  }
}

// Build genre chips and year preset chips under the Recommendations panel
async function initUserFilters() {
  // Genre chips populate the hidden text input #genres for compatibility
  const box = document.getElementById('user-genres');
  if (box) {
    try {
      const gsrc = await fetchJSON('/genres');
      let genres = Array.isArray(gsrc) ? gsrc.slice() : [];
      genres = genres.filter(g => g.toLowerCase() !== 'no genre listed' && g.toLowerCase() !== '(genres not listed)');
      genres.push('(genres not listed)');
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

  // Year preset chips write to #yearFrom and #yearTo inputs
  const yp = document.getElementById('year-presets');
  if (yp) {
    const presets = [
      {label: '1980s', from: 1980, to: 1989},
      {label: '1990s', from: 1990, to: 1999},
      {label: '2000s', from: 2000, to: 2009},
      {label: '2010s', from: 2010, to: 2019},
      {label: '2020s', from: 2020, to: 2029},
      {label: 'All', from: '', to: ''},
    ];
    presets.forEach((p) => {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'chip';
      b.textContent = p.label;
      b.onclick = () => {
        const yf = document.getElementById('yearFrom');
        const yt = document.getElementById('yearTo');
        if (yf) yf.value = p.from;
        if (yt) yt.value = p.to;
      };
      yp.appendChild(b);
    });
  }
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
      const posters = await postersFor(data);
      data.map((m) => ({...m, poster: posters[m.movieId]})).forEach((m) => card(row, m));
    } catch (e) {
      row.textContent = 'Search failed';
    }
  });
  // Enter to search
  $('#q').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); form.requestSubmit(); } });
}

window.addEventListener('DOMContentLoaded', async () => {
  await initUsersSelect();
  handleUserForm();
  initUserFilters();
  handleBrowse();
  await loadPopular();
  await loadMostClicked();
  await loadGenres();
  renderMyList();
  enableKeyboardScroll();
  // Restore last user
  const lastUser = localStorage.getItem('ml_user');
  if (lastUser && $('#userSelHeader')) { $('#userSelHeader').value = lastUser; }
  // View mode toggle: posters/list
  const savedMode = localStorage.getItem('ml_mode');
  if (savedMode === 'list') { document.body.classList.add('list-mode'); }
  const modeBtn = document.getElementById('mode-toggle');
  const syncIcon = () => { modeBtn.textContent = document.body.classList.contains('list-mode') ? '📄' : '🖼️'; };
  syncIcon();
  modeBtn.addEventListener('click', () => {
    document.body.classList.toggle('list-mode');
    const isList = document.body.classList.contains('list-mode');
    localStorage.setItem('ml_mode', isList ? 'list' : 'posters');
    syncIcon();
  });
  // TopN control
  initTopN();
});

// Favorites (localStorage)
const FAV_KEY = 'ml_favorites';
function getFavorites() {
  try { return new Set(JSON.parse(localStorage.getItem(FAV_KEY) || '[]')); } catch { return new Set(); }
}
function saveFavorites(set) {
  localStorage.setItem(FAV_KEY, JSON.stringify(Array.from(set)));
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
    const uid = localStorage.getItem('ml_user') || 'guest';
    await fetch('/feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ userId: uid, movieId, action }) });
  } catch {}
}
