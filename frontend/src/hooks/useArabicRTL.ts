// src/hooks/useArabicRTL.ts
import { useEffect } from 'react'

const ARABIC_RE = /[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/

function applyRTL(root: Element | Document = document) {
  const els = root.querySelectorAll<HTMLElement>(
    '.prose p, .prose li, .prose h1, .prose h2, .prose h3, .prose h4, .prose span, .prose td, .prose th'
  )
  els.forEach(el => {
    if (ARABIC_RE.test(el.textContent ?? '')) {
      el.dir = 'rtl'
    } else if (el.dir === 'rtl') {
      el.dir = ''   // reset if content changed
    }
  })
}

export function useArabicRTL() {
  useEffect(() => {
    applyRTL()

    const observer = new MutationObserver(mutations => {
      for (const m of mutations) {
        if (m.type === 'childList' || m.type === 'characterData') {
          applyRTL()
          break
        }
      }
    })

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      characterData: true,
    })

    return () => observer.disconnect()
  }, [])
}