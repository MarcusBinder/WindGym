/**
 * @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  tutorialSidebar: [
    // Getting Started
    "index",
    "installation",

    // Core Documentation
    {
      type: "category",
      label: "User Guide",
      collapsible: true,
      collapsed: false,
      items: [
        "concepts",
        "simulations",
        "agents",
        "evaluations",
        "noise-and-uncertainty",
      ],
    },

    // Reference
    {
      type: "category",
      label: "Reference",
      collapsible: true,
      collapsed: false,
      items: [
        "api-reference",
        {
          type: "category",
          label: "API (Auto-generated)",
          collapsible: true,
          collapsed: true,
          items: [
            "api/index",
            {
              type: "category",
              label: "Modules",
              collapsible: true,
              collapsed: true,
              items: [
                "api/modules/windgym",
                "api/modules/core",
                "api/modules/agents",
                "api/modules/wrappers",
                "api/modules/visualization",
                "api/modules/utils",
              ],
            },
          ],
        },
        "troubleshooting",
      ],
    },

    // Contributing
    "developer-guidelines",
  ],
};

module.exports = sidebars;
