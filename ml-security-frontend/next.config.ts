/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      {
        protocol: "https",
        hostname: "**", // allow all hosts
      },
    ],
  },

  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://localhost:5000/:path*', // 👈 promijeni u URL svog backend servera
      },
    ]
  },
}

export default nextConfig
