import type { Metadata } from 'next'
import '@fontsource/roboto/300.css'
import '@fontsource/roboto/400.css'
import '@fontsource/roboto/500.css'
import '@fontsource/roboto/700.css'
import './globals.css'

export const metadata: Metadata = {
  title: 'Folkdevils AI',
  description: 'Next.js application with modern tech stack',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="antialiased font-roboto">
        {children}
      </body>
    </html>
  )
}
