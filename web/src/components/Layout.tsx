import { Link } from 'react-router-dom'

export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <>
      <header className="site-header">
        <div className="shell">
          <Link to="/" className="brand">
            <img src="/favicon.svg" alt="" />
            Predicting Prediction Markets
          </Link>
        </div>
      </header>
      <main className="shell">{children}</main>
      <footer className="site-footer">
        <div className="shell">
          <p>
            Models trained on resolved Polymarket markets. Every prediction is stored and checked
            against the real outcome once the market resolves.
          </p>
          <p>A research project, not financial advice.</p>
        </div>
      </footer>
    </>
  )
}
