import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Link, Route, Routes } from 'react-router-dom'
import './index.css'
import Layout from './components/Layout'
import Home from './pages/Home'
import Market from './pages/Market'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/market/:marketId" element={<Market />} />
          <Route path="*" element={<p>Page not found. <Link to="/">Search for a market</Link></p>} />
        </Routes>
      </Layout>
    </BrowserRouter>
  </StrictMode>,
)
