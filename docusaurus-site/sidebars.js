/**
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  tutorialSidebar: [
    // The 'intro' page (index.md) should usually be first
    'index', // Refers to docusaurus-site/docs/index.md

    // Then, list your other pages in the order you want them to appear
    'installation', // Refers to docusaurus-site/docs/installation.md
    'concepts',     // Refers to docusaurus-site/docs/concepts.md
    'agents',       // Refers to docusaurus-site/docs/agents.md
    'evaluations',  // Refers to docusaurus-site/docs/evaluations.md

    // If you have notebooks converted to Markdown, list them here too
    // For example, if simulations.ipynb converts to simulations.md:
    // 'simulations', // Refers to docusaurus-site/docs/simulations.md

    // You can also group related pages into categories
    /*
    {
      type: 'category',
      label: 'Advanced Topics',
      items: [
        'advanced/topic1',
        'advanced/topic2',
      ],
    },
    */
  ],
};

module.exports = sidebars;
