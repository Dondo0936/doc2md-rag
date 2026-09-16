import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { PlatformShell } from "@/components/PlatformShell";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Doc2MD Platform — Parser",
  description:
    "Parser module: upload, paste, or fetch documents into structured blocks.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${inter.variable} ${jetbrainsMono.variable} h-full antialiased`}
    >
      <body className="min-h-full font-sans">
        <PlatformShell>{children}</PlatformShell>
      </body>
    </html>
  );
}
