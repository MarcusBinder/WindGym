// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion
const { themes } = require('prism-react-renderer');
const lightCodeTheme = themes.oceanicNext;
const darkCodeTheme = themes.nightOwl;
/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Hello WindGym Docs',
  tagline: 'Just a simple landing page',
  url: 'https://pages.windenergy.dtu.dk',
  baseUrl: '/',
  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  favicon: 'img/favicon.ico',
  // GitHub pages deployment config.
  organizationName: 'DTU Wind Energy Systems',
  projectName: 'WindGym',
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },
  presets: [
    [
      '@docusaurus/preset-classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: '/', // Serve docs from the root
          sidebarPath: require.resolve('./sidebars.js'),
          editUrl:
            'https://gitlab.windenergy.dtu.dk/sys/windgym/dino_docs/-/edit/main/docusaurus-site',
          // === TEMPORARILY REMOVE ALL remarkPlugins and rehypePlugins FOR TESTING ===
          // remarkPlugins: [], // No plugins for now
          // rehypePlugins: [], // No plugins for now
        },
        blog: false,
        theme: {
          customCss: require.resolve('./src/css/custom.css'),
        },
      }),
    ],
  ],
  // === TEMPORARILY REMOVE MATH STYLESHEET FOR TESTING ===
  stylesheets: [
    // {
    //   href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
    //   type: 'text/css',
    //   integrity: 'sha384-Tw7uHKUwpYad+s7fdlA0PlsHQJpgbFVgYwNwuS/nD6WbOWbKijxX+v+rUtyzT6b0',
    //   crossorigin: 'anonymous',
    // },
  ],
  // === END MATH SUPPORT ADDITION ===
  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      navbar: {
        title: 'Hello WindGym Docs',
        items: [],
      },
      footer: {
        style: 'dark',
        links: [],
        copyright: `Copyright © ${new Date().getFullYear()} WindGym. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
};
module.exports = config;
