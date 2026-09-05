import './globals.css';
export const metadata = {
  title: 'simv1 · Room reconstruction',
  description:
    'Independent room captures, inspectable alignment, shared geometry.',
};
export default function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
