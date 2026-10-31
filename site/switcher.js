// The world switcher on each example page: a cube that opens a menu of the other worlds and the home page.
// <script src="../switcher.js" data-current="city" data-phone="top"></script>
// data-phone: where it sits on phones, 'top' (default) or 'bottom'. On wider screens it sits bottom left.
;(() => {
    const script = document.currentScript
    const current = script?.dataset.current ?? ''
    const phone = script?.dataset.phone === 'bottom' ? 'bottom' : 'top'
    const base = new URL('./', script?.src ?? location.href).href
    const worlds = [
        {id: 'city', title: 'White city', href: 'city/', color: '#ef3b36'},
        {id: 'bodies', title: 'Seven bodies', href: 'bodies/', color: '#5b3ce0'},
        {id: 'playground', title: 'Playground', href: 'playground/', color: '#f2a531'},
        {id: 'plain', title: 'Plain three.js', href: 'plain/', color: '#9de02c'},
    ]
    const name = worlds.find(w => w.id === current)?.title ?? 'three-motionbricks'
    const cube = (s = 18) => `<svg width="${s}" height="${s * 20 / 18}" viewBox="0 0 18 20" aria-hidden="true"><polygon points="9,1 17,5.5 9,10 1,5.5" fill="#7b5ff0"/><polygon points="1,5.5 9,10 9,19 1,14.5" fill="#5b3ce0"/><polygon points="17,5.5 9,10 9,19 17,14.5" fill="#4428b8"/></svg>`

    const style = document.createElement('style')
    style.textContent = `
.mbs { position: fixed; z-index: 1000; left: max(16px, env(safe-area-inset-left, 0px)); bottom: calc(16px + env(safe-area-inset-bottom, 0px)); font: 700 12px/1 'Space Mono', ui-monospace, monospace; color: #16171a; }
.mbs summary { display: flex; align-items: center; gap: 8px; min-height: 40px; padding: 0 12px; list-style: none; cursor: pointer; background: #fff; border: 2px solid #16171a; box-shadow: 4px 4px 0 #16171a; user-select: none; }
.mbs summary::-webkit-details-marker { display: none; }
.mbs summary:focus-visible, .mbs a:focus-visible { outline: 3px solid #5b3ce0; outline-offset: 2px; }
.mbs .lbl { color: #5c5e66; font-weight: 400; }
.mbs .caret { transition: transform 0.15s; }
.mbs[open] .caret { transform: rotate(180deg); }
.mbs ul { position: absolute; left: 0; bottom: calc(100% + 10px); min-width: 210px; margin: 0; padding: 6px 0; list-style: none; background: #fff; border: 2px solid #16171a; box-shadow: 4px 4px 0 #16171a; }
.mbs li a { display: flex; align-items: center; gap: 10px; padding: 10px 14px; color: #16171a; text-decoration: none; }
.mbs li a:hover { background: #f1f1f3; }
.mbs li a[aria-current="page"] { background: #16171a; color: #fff; }
.mbs li .dot { width: 10px; height: 10px; flex: none; }
.mbs li.home { border-top: 1px solid #e4e4e8; margin-top: 6px; padding-top: 6px; }
@media (max-width: 560px) {
  .mbs { left: max(12px, env(safe-area-inset-left, 0px)); ${phone === 'top' ? 'top: calc(12px + env(safe-area-inset-top, 0px)); bottom: auto;' : 'bottom: calc(12px + env(safe-area-inset-bottom, 0px));'} }
  .mbs summary { padding: 0 10px; box-shadow: 3px 3px 0 #16171a; }
  .mbs summary .txt { display: none; }
  ${phone === 'top' ? '.mbs ul { bottom: auto; top: calc(100% + 10px); }' : ''}
}
@media (prefers-reduced-motion: reduce) { .mbs .caret { transition: none; } }`
    document.head.appendChild(style)

    const nav = document.createElement('details')
    nav.className = 'mbs'
    nav.innerHTML = `<summary aria-label="Worlds: ${name}">${cube()}<span class="txt">motionbricks <span class="lbl">/</span> ${name}</span><span class="caret" aria-hidden="true">▴</span></summary>
<ul>${worlds.map(w => `<li><a href="${base + w.href}"${w.id === current ? ' aria-current="page"' : ''}><span class="dot" style="background:${w.color}"></span>${w.title}</a></li>`).join('')}
<li class="home"><a href="${base}">${cube(14)} home</a></li></ul>`
    // Close when clicking elsewhere, or on Escape.
    addEventListener('pointerdown', e => { if (nav.open && !nav.contains(e.target)) nav.open = false })
    addEventListener('keydown', e => { if (e.key === 'Escape' && nav.open) { nav.open = false; nav.querySelector('summary').focus() } })
    const add = () => document.body.appendChild(nav)
    document.body ? add() : addEventListener('DOMContentLoaded', add)
})()
