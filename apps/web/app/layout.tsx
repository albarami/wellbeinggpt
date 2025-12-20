import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "المُحاسبي - منصة العالِم",
  description: "حوار علمي مؤصّل مع أدلة وتتبّع للمسار الاستدلالي",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ar" dir="rtl">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link 
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+Arabic:wght@400;500;600;700&display=swap" 
          rel="stylesheet"
        />
      </head>
      <body className="min-h-screen">
        <div className="min-h-screen flex flex-col">
          {/* Header */}
          <header className="sticky top-0 z-50 glass border-b border-slate-200/50">
            <div className="mx-auto max-w-[1600px] px-6 py-4">
              <div className="flex items-center justify-between">
                {/* Logo & Brand */}
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-500 to-teal-700 flex items-center justify-center shadow-lg">
                    <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
                    </svg>
                  </div>
                  <div>
                    <h1 className="text-lg font-bold text-slate-800">
                      المُحاسبي
                    </h1>
                    <p className="text-xs text-slate-500">منصة العالِم للحياة الطيبة</p>
                  </div>
                </div>

                {/* Navigation */}
                <nav className="flex items-center gap-2">
                  <a 
                    href="/chat" 
                    className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium text-slate-600 hover:bg-white hover:text-teal-600 hover:shadow-md transition-all duration-200"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                    </svg>
                    الحوار
                  </a>
                  <a 
                    href="/graph" 
                    className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-medium text-slate-600 hover:bg-white hover:text-teal-600 hover:shadow-md transition-all duration-200"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
                    </svg>
                    الخريطة المعرفية
                  </a>
                </nav>
              </div>
            </div>
          </header>

          {/* Main Content */}
          <main className="flex-1 mx-auto w-full max-w-[1600px] px-6 py-6">
            {children}
          </main>

          {/* Footer */}
          <footer className="border-t border-slate-200/50 bg-white/50">
            <div className="mx-auto max-w-[1600px] px-6 py-4">
              <div className="flex items-center justify-between text-xs text-slate-400">
                <span>© ٢٠٢٤ منصة المُحاسبي للحياة الطيبة</span>
                <span>إصدار تجريبي · جميع الإجابات مؤصّلة من الإطار</span>
              </div>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
