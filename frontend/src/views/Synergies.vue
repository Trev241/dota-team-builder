<template>
  <div class="mt-12">
    <div class="max-w-6xl mx-auto">
      <h1 class="text-4xl font-bold mb-4">Hero Embeddings</h1>

      <p class="mb-4">
        Here is a visual representation of the hero embeddings that were generated during training.
        The embeddings you see below have been reduced from 128 dimensions to 3 dimensions using PCA
        and t-SNE.
      </p>

      <p class="mb-4">
        Heroes located closer together in the embedding space are said to have good synergy in
        theory.
      </p>

      <div>
        <h2 class="text-2xl mt-12 font-semibold text-center">3D Embeddings</h2>
        <p class="mb-4 text-center">An interactive plot visualizing hero embeddings in 3D space</p>
        <div class="flex flex-row justify-center mb-4">
          <div v-for="role in roles" :key="role.id" class="flex items-center mr-6">
            <span
              :style="`background:${role.color}`"
              class="w-4 h-4 rounded-full mr-2 border"
            ></span>
            <span class="text-gray-300">{{ role.name }}</span>
          </div>
        </div>
        <div class="relative w-full h-[600px]">
          <div
            v-show="isLoading"
            class="absolute inset-0 bg-gray-800 animate-pulse rounded z-10"
          ></div>
          <div ref="plot3dContainer" class="w-full h-full"></div>
        </div>
      </div>

      <div>
        <h2 class="text-2xl font-semibold text-center mt-12">2D Embeddings</h2>
        <p class="mb-4 text-center">To see a hero's name, hover over the marker</p>
        <div class="flex flex-row justify-center mb-4">
          <div v-for="role in roles" :key="role.id" class="flex items-center mr-6">
            <span
              :style="`background:${role.color}`"
              class="w-4 h-4 rounded-full mr-2 border"
            ></span>
            <span class="text-gray-300">{{ role.name }}</span>
          </div>
        </div>
        <div class="relative w-full h-[600px] mb-4">
          <div
            v-show="isLoading"
            class="absolute inset-0 bg-gray-800 animate-pulse rounded z-10"
          ></div>
          <div ref="plot2dContainer" class="w-full h-full"></div>
        </div>
        <p class="mb-4">
          Observe how roles do not form clusters. This makes intuitive sense actually. If heroes
          closer together in an embedding space are synergestic, then it's obvious that we wouldn't
          expect to see, for example, three hard supports placed close together. They would instead
          be better of complementing heroes belonging to other roles which is what we see in this
          embedding chart.
        </p>
      </div>

      <div>
        <h2 class="text-4xl font-bold mt-12 mb-4">Synergy Graph</h2>
        <p class="mb-4">
          The charts above do not explicitly produce good synergy pairs. So instead of just plotting
          the embeddings in a chart, we can produce a synergy matrix produced by taking the cosine
          similarities of different hero pairs. Using this synergy matrix, we can then construct a
          network graph.
        </p>

        <p class="mb-4">
          Each node represnts a hero. For every pair of heroes, an edge is created if and only if
          the synergy between the two heroes is above a certain threshold. This makes it so that we
          only highlight noteworthy synergies while ignoring those that could be possibly negligible
          or even anti-synergestic.
        </p>

        <iframe src="/hero_synergy_graph.html" class="w-full h-[600px] border-0"></iframe>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from "vue"
import Plotly from "plotly.js-dist-min"

const plot3dContainer = ref(null)
const plot2dContainer = ref(null)
const isLoading = ref(true)

const roles = [
  { id: 0, name: "Carry", color: "red" },
  { id: 1, name: "Mid", color: "blue" },
  { id: 2, name: "Offlane", color: "green" },
  { id: 3, name: "Support", color: "orange" },
  { id: 4, name: "Hard Support", color: "purple" },
]

const roleColors = {
  0: "red",
  1: "blue",
  2: "green",
  3: "orange",
  4: "purple",
}

const createEmbeddingsPlot3d = async () => {
  const response = await fetch("/embeddings_3d.json")
  const embeddings = await response.json()

  const x = []
  const y = []
  const z = []
  const text = []
  const colors = []

  for (const embedding of embeddings) {
    x.push(embedding.embedding[0])
    y.push(embedding.embedding[1])
    z.push(embedding.embedding[2])
    text.push(embedding.localized_name)
    colors.push(roleColors[embedding.role])
  }

  const trace = {
    x: x,
    y: y,
    z: z,
    mode: "markers+text",
    text: text,
    textposition: "right",
    marker: {
      size: 14,
      color: colors,
      line: { color: "#222", width: 1 },
      opacity: 0.85,
    },
    textfont: {
      size: 8,
      color: "#111",
    },
    hoverlabel: {
      bgcolor: "#edf2f7",
      font: { color: "#1a202c", size: 13, family: "Inter, Arial, sans-serif" },
    },
    type: "scatter3d",
  }

  const layout = {
    paper_bgcolor: "#f7fafc",
    plot_bgcolor: "#ffffff",
    margin: { l: 0, r: 0, b: 0, t: 40 },
    scene: {
      camera: {
        eye: { x: 0.75, y: 0.75, z: 0.05 },
      },
      xaxis: { title: "X Axis", titlefont: { size: 16, color: "#1a202c" } },
      yaxis: { title: "Y Axis", titlefont: { size: 16, color: "#1a202c" } },
      zaxis: { title: "Z Axis", titlefont: { size: 16, color: "#1a202c" } },
    },
    font: { family: "Inter, Arial, sans-serif", size: 12, color: "#4a5568" },
  }

  await Plotly.newPlot(plot3dContainer.value, [trace], layout)
}

const createEmbeddingsPlot2d = async () => {
  const response = await fetch("/embeddings_2d.json")
  const embeddings = await response.json()

  const x = []
  const y = []
  const names = []
  const colors = []

  for (const embedding of embeddings) {
    x.push(embedding.embedding[0])
    y.push(embedding.embedding[1])
    names.push(embedding.localized_name)
    colors.push(roleColors[embedding.role])
  }

  const trace = {
    x: x,
    y: y,
    mode: "markers",
    text: names,
    textposition: "top center",
    marker: {
      size: 12,
      color: colors,
      opacity: 0.85,
    },
    textfont: {
      size: 8,
      color: "#111",
    },
    hoverlabel: {
      bgcolor: "#edf2f7",
      font: { color: "#1a202c", size: 13, family: "Inter, Arial, sans-serif" },
    },
  }

  const layout = {
    title: "Hero Embeddings 2D Projection",
    xaxis: { title: "PC 1" },
    yaxis: { title: "PC 2" },
    plot_bgcolor: "#fff",
    paper_bgcolor: "#f9fafb",
    font: { family: "Arial, sans-serif", size: 12, color: "#333" },
  }

  await Plotly.newPlot(plot2dContainer.value, [trace], layout, { responsive: true })
}

onMounted(async () => {
  await createEmbeddingsPlot3d()
  await createEmbeddingsPlot2d()

  isLoading.value = false
})
</script>
