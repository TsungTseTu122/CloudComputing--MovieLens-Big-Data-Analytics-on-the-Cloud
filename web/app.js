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
  row.innerHTML = '';
  try {
    const data = await fetchJSON('/popular?topN=50');
    const posters = await postersFor(data);
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
    const genres = await fetchJSON('/genres');
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
  } catch (e) {
    list.textContent = 'No genres available';
  }
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
    const userId = $('#userId').value.trim();
    const topN = Number($('#topN').value || 20);
    const genres = $('#genres').value.trim();
    const yearFrom = $('#yearFrom').value.trim();
    const yearTo = $('#yearTo').value.trim();
    const row = $('#user-row');
    row.innerHTML = '';
    if (!userId) {
      row.textContent = 'Enter a userId to see personalized recommendations';
      return;
    }
    const params = new URLSearchParams({ topN: String(topN) });
    if (genres) params.set('genres', genres);
    if (yearFrom) params.set('year_from', yearFrom);
    if (yearTo) params.set('year_to', yearTo);
    try {
      const data = await fetchJSON(`/recommendations/user/${encodeURIComponent(userId)}?${params}`);
      const posters = await postersFor(data);
      const withPosters = data.map((m) => ({...m, poster: posters[m.movieId]}));
      if (!withPosters.length) {
        row.textContent = 'No recommendations found for this user.';
      } else {
        withPosters.forEach((m) => card(row, m));
      }
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

function handleBrowse() {
  const form = $('#browse-form');
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const q = $('#q').value.trim();
    const topN = Number($('#bTopN').value || 50);
    const row = $('#browse-row');
    row.innerHTML = '';
    try {
      const data = await fetchJSON(`/movies?topN=${topN}${q ? `&q=${encodeURIComponent(q)}` : ''}`);
      const posters = await postersFor(data);
      data.map((m) => ({...m, poster: posters[m.movieId]})).forEach((m) => card(row, m));
    } catch (e) {
      row.textContent = 'Search failed';
    }
  });
}

window.addEventListener('DOMContentLoaded', async () => {
  handleUserForm();
  handleBrowse();
  await loadPopular();
  await loadGenres();
  renderMyList();
  enableKeyboardScroll();
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
  if (!ids.length) { row.textContent = 'Add titles with the heart to build your list.'; return; }
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
