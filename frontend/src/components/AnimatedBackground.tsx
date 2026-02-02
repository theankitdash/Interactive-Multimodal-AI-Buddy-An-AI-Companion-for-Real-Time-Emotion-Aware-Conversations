import { useEffect, useRef } from 'react'

export default function AnimatedBackground() {
    const canvasRef = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        const canvas = canvasRef.current
        if (!canvas) return

        const ctx = canvas.getContext('2d')
        if (!ctx) return

        // Get particle color from CSS variable
        const computedStyle = getComputedStyle(document.documentElement)
        const particleColor = computedStyle.getPropertyValue('--color-particle').trim() || '187, 134, 252'

        // Set canvas size
        const resizeCanvas = () => {
            canvas.width = window.innerWidth
            canvas.height = window.innerHeight
        }
        resizeCanvas()
        window.addEventListener('resize', resizeCanvas)

        // Particles array
        const particles: Array<{
            x: number
            y: number
            vx: number
            vy: number
            size: number
            opacity: number
        }> = []

        // Create particles
        const particleCount = 60
        for (let i = 0; i < particleCount; i++) {
            particles.push({
                x: Math.random() * canvas.width,
                y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.5,
                vy: (Math.random() - 0.5) * 0.5,
                size: Math.random() * 2 + 1,
                opacity: Math.random() * 0.5 + 0.2
            })
        }

        let animationId: number

        // Animation loop
        const animate = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height)

            // Draw and update particles
            particles.forEach((particle, i) => {
                // Update position
                particle.x += particle.vx
                particle.y += particle.vy

                // Wrap around edges
                if (particle.x < 0) particle.x = canvas.width
                if (particle.x > canvas.width) particle.x = 0
                if (particle.y < 0) particle.y = canvas.height
                if (particle.y > canvas.height) particle.y = 0

                // Draw particle
                ctx.beginPath()
                ctx.arc(particle.x, particle.y, particle.size, 0, Math.PI * 2)
                ctx.fillStyle = `rgba(${particleColor}, ${particle.opacity})`
                ctx.fill()

                // Draw connections
                particles.slice(i + 1).forEach(otherParticle => {
                    const dx = particle.x - otherParticle.x
                    const dy = particle.y - otherParticle.y
                    const distance = Math.sqrt(dx * dx + dy * dy)

                    if (distance < 150) {
                        ctx.beginPath()
                        ctx.moveTo(particle.x, particle.y)
                        ctx.lineTo(otherParticle.x, otherParticle.y)
                        ctx.strokeStyle = `rgba(${particleColor}, ${0.1 * (1 - distance / 150)})`
                        ctx.lineWidth = 1
                        ctx.stroke()
                    }
                })
            })

            animationId = requestAnimationFrame(animate)
        }

        animate()

        return () => {
            window.removeEventListener('resize', resizeCanvas)
            cancelAnimationFrame(animationId)
        }
    }, [])

    return (
        <canvas
            ref={canvasRef}
            className="animated-background"
            style={{
                position: 'fixed',
                top: 0,
                left: 0,
                width: '100%',
                height: '100%',
                zIndex: 1,
                pointerEvents: 'none'
            }}
        />
    )
}

