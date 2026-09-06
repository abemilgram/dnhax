import './globals.css';
export const metadata = {
  title: 'DNHacks Defense · Tactical Brain',
  description:
    'Deterministic tactical replay with a separate reconstruction lab.',
};
export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
