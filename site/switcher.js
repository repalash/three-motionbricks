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
        {id: 'park', title: 'Park', href: 'park/', color: '#2e9d5b'},
        {id: 'tag', title: 'Tag', href: 'tag/', color: '#ef3b36'},
        {id: 'plain', title: 'Plain three.js', href: 'plain/', color: '#9de02c'},
    ]
    const name = worlds.find(w => w.id === current)?.title ?? 'three-motionbricks'
    const cube = (s = 18) => `<svg width="${s}" height="${s * 20 / 18}" viewBox="0 0 18 20" aria-hidden="true"><polygon points="9,1 17,5.5 9,10 1,5.5" fill="#7b5ff0"/><polygon points="1,5.5 9,10 9,19 1,14.5" fill="#5b3ce0"/><polygon points="17,5.5 9,10 9,19 17,14.5" fill="#4428b8"/></svg>`

    const style = document.createElement('style')
    style.textContent = `
.mbs-bar { position: fixed; z-index: 1000; left: max(16px, env(safe-area-inset-left, 0px)); bottom: calc(16px + env(safe-area-inset-bottom, 0px)); display: flex; align-items: flex-end; gap: 10px; }
.mbs { position: relative; font: 700 12px/1 'Space Mono', ui-monospace, monospace; color: #16171a; }
.mbs-pause { display: flex; align-items: center; gap: 8px; min-height: 40px; padding: 0 12px; font: 700 12px/1 'Space Mono', ui-monospace, monospace; color: #16171a; background: #fff; border: 2px solid #16171a; box-shadow: 4px 4px 0 #16171a; cursor: pointer; }
.mbs-pause:focus-visible { outline: 3px solid #5b3ce0; outline-offset: 2px; }
.mbs-pause[aria-pressed="true"] { color: #fff; background: #16171a; box-shadow: 4px 4px 0 #f2a531; }
.mbs-pause kbd { padding: 0 4px; border: 1.5px solid currentColor; font: inherit; font-size: 10px; line-height: 14px; }
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
  .mbs-bar { left: max(12px, env(safe-area-inset-left, 0px)); ${phone === 'top' ? 'top: calc(12px + env(safe-area-inset-top, 0px)); bottom: auto; align-items: flex-start;' : 'bottom: calc(12px + env(safe-area-inset-bottom, 0px));'} }
  .mbs summary, .mbs-pause { padding: 0 10px; box-shadow: 3px 3px 0 #16171a; }
  .mbs-pause kbd, .mbs-pause .txt { display: none; }
  .mbs summary .txt { display: none; }
  ${phone === 'top' ? '.mbs ul { bottom: auto; top: calc(100% + 10px); }' : ''}
}
@media (prefers-reduced-motion: reduce) { .mbs .caret { transition: none; } }`
    document.head.appendChild(style)

    // Pause: the button or P stops the motion, and the physics in the games, while the camera still orbits, for a look
    // at the pose. Pages read window.motionbricksPaused each frame; whichever script comes first sets this up.
    const pause = window.motionbricksPause ??= (() => {
        const api = {
            paused: false,
            toggle(value = !api.paused) {
                api.paused = window.motionbricksPaused = value
                dispatchEvent(new CustomEvent('motionbricks:pause', {detail: value}))
            },
        }
        addEventListener('keydown', e => {
            if (e.code === 'KeyP' && !e.repeat && !e.ctrlKey && !e.metaKey && !e.altKey && !/^(INPUT|TEXTAREA|SELECT)$/.test(e.target.tagName)) api.toggle()
        })
        return api
    })()
    const button = document.createElement('button')
    button.type = 'button'
    button.className = 'mbs-pause'
    const label = () => {
        button.setAttribute('aria-pressed', String(pause.paused))
        button.innerHTML = pause.paused ? '<span aria-hidden="true">▶</span><span class="txt">play</span><kbd>P</kbd>' : '<span aria-hidden="true">❚❚</span><span class="txt">pause</span><kbd>P</kbd>'
        button.setAttribute('aria-label', pause.paused ? 'Play' : 'Pause')
    }
    label()
    button.addEventListener('click', () => pause.toggle())
    addEventListener('motionbricks:pause', label)

    const nav = document.createElement('details')
    nav.className = 'mbs'
    nav.innerHTML = `<summary aria-label="Worlds: ${name}">${cube()}<span class="txt">motionbricks <span class="lbl">/</span> ${name}</span><span class="caret" aria-hidden="true">▴</span></summary>
<ul>${worlds.map(w => `<li><a href="${base + w.href}"${w.id === current ? ' aria-current="page"' : ''}><span class="dot" style="background:${w.color}"></span>${w.title}</a></li>`).join('')}
<li class="home"><a href="${base}">${cube(14)} home</a></li></ul>`
    // Close when clicking elsewhere, or on Escape.
    addEventListener('pointerdown', e => { if (nav.open && !nav.contains(e.target)) nav.open = false })
    addEventListener('keydown', e => { if (e.key === 'Escape' && nav.open) { nav.open = false; nav.querySelector('summary').focus() } })
    const bar = document.createElement('div')
    bar.className = 'mbs-bar'
    bar.append(nav)
    // Not on the home page, which has nothing moving to pause.
    if (current) bar.append(button)
    const add = () => document.body.appendChild(bar)
    document.body ? add() : addEventListener('DOMContentLoaded', add)
})()
