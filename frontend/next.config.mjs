/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  eslint: { ignoreDuringBuilds: false },
  async rewrites() {
    // Lets the browser call /api/* on the same origin in development.
    const base = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
    return [
      { source: "/backend-api/:path*", destination: `${base}/api/:path*` },
    ];
  },
};

export default nextConfig;
