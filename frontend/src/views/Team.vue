<template>
  <div class="relative min-h-screen mx-auto p-6 font-sans">
    <!-- Overlay text on large screens -->
    <div
      v-if="!isSmallScreen && typedText"
      class="fixed inset-0 flex justify-center text-white uppercase items-center font-bold text-8xl select-none pointer-events-none drop-shadow-[0_1.2px_1.2px_rgba(0,0,0,0.8)] z-50"
    >
      {{ typedText }}
    </div>

    <h1 class="text-4xl font-bold mb-2">Assemble a team</h1>
    <p class="text-xl mb-6">Select a few heroes to get started.</p>

    <h2 class="text-xl text-center font-semibold mb-4">Your Team</h2>
    <div class="flex flex-wrap max-w-4xl mx-auto mb-8">
      <div
        v-for="index in maxTeamSize"
        :key="index"
        @click="removeHero(team[index - 1])"
        class="w-1/2 md:w-1/5 rounded-lg flex flex-col items-center justify-center text-gray-400"
      >
        <template v-if="team[index - 1]">
          <div class="hover:bg-red-100 p-2 rounded-lg">
            <img
              :src="team[index - 1].icon_url"
              :alt="team[index - 1].localized_name"
              class="object-contain rounded-md"
            />
            <div class="mt-1 font-semibold text-gray-900 text-center">
              {{ team[index - 1].localized_name }}
            </div>
          </div>
        </template>
        <template v-else>
          <div class="flex items-center h-40">
            <p class="text-lg italic">Empty Slot</p>
          </div>
        </template>
      </div>
      <!-- <div
        v-for="hero in team"
        @click="removeHero(hero)"
        :key="hero.id"
        class="w-1/2 md:w-1/5 hover:bg-red-100 rounded-lg p-2 flex flex-col items-center hover:shadow-lg transition-shadow relative"
      >
        <img
          :src="hero.icon_url"
          :alt="hero.localized_name"
          class="w-full rounded-md object-contain"
        />
        <div class="text-center font-semibold mt-2">{{ hero.localized_name }}</div>
      </div> -->
    </div>

    <h2 class="text-xl font-semibold">Available Heroes</h2>
    <p v-if="!isSmallScreen" class="mb-4">Type in anywhere to search for a hero</p>
    <input
      v-if="isSmallScreen"
      v-model="searchQuery"
      placeholder="Search heroes"
      type="text"
      class="w-full p-3 mt-2 mb-6 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
    />
    <div class="flex flex-wrap">
      <div
        v-for="hero in filteredAndSortedHeroes"
        :key="hero.id"
        :class="[
          'w-1/3 md:w-30 rounded-lg hover:bg-green-200 p-1 flex flex-col items-center cursor-pointer hover:shadow-lg transition-shadow',
          team.length >= 5 ? 'opacity-50 pointer-events-none' : 'bg-white',
          !hero.filtered && 'opacity-10',
        ]"
        @click="addHero(hero)"
        :aria-disabled="team.length >= 5"
        :title="team.length >= 5 ? 'Team is full' : hero.localized_name"
      >
        <img
          :src="hero.icon_url"
          :alt="hero.localized_name"
          class="w-full rounded-md object-contain"
        />
        <!-- <div class="text-center font-semibold mt-1">{{ hero.localized_name }}</div> -->
      </div>
      <div v-if="filteredAndSortedHeroes.length === 0" class="text-gray-500 mt-4">
        No heroes match your search.
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, hydrateOnIdle } from "vue"

const maxTeamSize = 5

const searchQuery = ref("")
const typedText = ref("")
const debounceTimeout = ref(null)
const isSmallScreen = ref(window.innerWidth < 768)
const team = ref([])
const heroes = ref([]) // initially empty, to be fetched

function resetTypedTextDebounced() {
  if (debounceTimeout.value) clearTimeout(debounceTimeout.value)
  debounceTimeout.value = setTimeout(() => {
    typedText.value = ""
    searchQuery.value = typedText.value
  }, 3500)
}

async function onGlobalKeyDown(event) {
  // Ignore on smaller devices
  if (isSmallScreen.value) return

  if (event.key.length === 1 && !event.ctrlKey && !event.metaKey) {
    typedText.value += event.key
    searchQuery.value = typedText.value
  } else if (event.key === "Backspace") {
    typedText.value = typedText.value.slice(0, -1)
    searchQuery.value = typedText.value
  } else if (event.key === "Enter") {
    const filteredHeroes = availableHeroes.value.filter((hero) =>
      hero.localized_name.toLowerCase().includes(searchQuery.value.toLowerCase()),
    )
    await addHero(filteredHeroes[0])
    searchQuery.value = typedText.value = ""
  }

  resetTypedTextDebounced()
}

function onResize() {
  isSmallScreen.value = window.innerWidth < 768
  if (isSmallScreen.value) typedText.value = ""
}

onMounted(async () => {
  window.addEventListener("keydown", onGlobalKeyDown)
  window.addEventListener("resize", onResize)

  const response = await fetch("/heroes.json")
  heroes.value = await response.json()
})

onUnmounted(() => {
  window.removeEventListener("keydown", onGlobalKeyDown)
  window.removeEventListener("resize", onResize)
  if (debounceTimeout.value) clearTimeout(debounceTimeout.value)
})

const availableHeroes = computed(() =>
  heroes.value.filter((h) => !team.value.some((th) => th.id === h.id)),
)

const filterHeroIcons = () => {
  const filtered = []

  for (const hero of availableHeroes.value) {
    const isQueryMatched = hero.localized_name
      .toLowerCase()
      .includes(searchQuery.value.toLowerCase())

    if (isQueryMatched || !isSmallScreen.value) {
      const heroData = { ...hero }
      if ((isQueryMatched && !isSmallScreen.value) || isSmallScreen.value) heroData.filtered = true
      filtered.push(heroData)
    }
  }

  return filtered.sort((a, b) => {
    if (a.score && b.score) return b.score - a.score
    else return a.localized_name.localeCompare(b.localized_name)
  })
}

const filteredAndSortedHeroes = computed(() => filterHeroIcons())

async function addHero(hero) {
  if (team.value.length >= 5 || team.value.some((h) => h.id === hero.id)) return
  team.value.push(hero)

  try {
    const response = await fetch("http://localhost:8000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ heroes: team.value.map((hero) => hero.id) }),
    })

    if (!response.ok) throw new Error("Prediction API error")

    const results = await response.json()
    heroes.value = results.sorted_candidates || []
  } catch (error) {
    console.error(error)
    heroes.value = []
  }
}

function removeHero(hero) {
  team.value = team.value.filter((h) => h.id !== hero.id)
}
</script>
