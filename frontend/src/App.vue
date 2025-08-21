<template>
  <div class="relative min-h-screen mx-auto p-6 font-sans">
    <!-- Overlay text on large screens -->
    <div
      v-if="!isSmallScreen && typedText"
      class="fixed inset-0 flex justify-center text-white uppercase items-center font-bold text-8xl select-none pointer-events-none drop-shadow-[0_1.2px_1.2px_rgba(0,0,0,0.8)] z-50"
    >
      {{ typedText }}
    </div>

    <h1 class="text-4xl font-bold mb-2">Dota 2 Team Builder</h1>
    <p class="text-xl mb-6">Select heroes to create your new team.</p>

    <h2 class="text-xl font-semibold mb-4">Your Team</h2>
    <div class="flex flex-wrap mb-8">
      <div
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
      </div>
    </div>

    <h2 class="text-xl font-semibold mb-4">Available Heroes</h2>
    <input
      v-if="isSmallScreen"
      v-model="searchQuery"
      placeholder="Search heroes"
      type="text"
      class="w-full p-3 mb-6 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
    />
    <div class="flex flex-wrap">
      <div
        v-for="hero in filteredAndSortedHeroes"
        :key="hero.id"
        :class="[
          'w-1/3 md:w-30 rounded-lg p-1 flex flex-col items-center cursor-pointer hover:shadow-lg transition-shadow',
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

function onGlobalKeyDown(event) {
  // Ignore on smaller devices
  if (isSmallScreen.value) return

  console.log(event.key)

  if (event.key.length === 1 && !event.ctrlKey && !event.metaKey) {
    typedText.value += event.key
    searchQuery.value = typedText.value
  } else if (event.key === "Backspace") {
    typedText.value = typedText.value.slice(0, -1)
    searchQuery.value = typedText.value
  } else if (event.key === "Enter") {
    addHero(filteredAndSortedHeroes[0])
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

const filteredAndSortedHeroes = computed(() => {
  const filtered = []
  for (const hero of availableHeroes.value) {
    const isQueryMatched = hero.localized_name
      .toLowerCase()
      .includes(searchQuery.value.toLowerCase())
    if (isQueryMatched || !isSmallScreen.value) {
      const heroData = { ...hero }
      if (isQueryMatched && !isSmallScreen.value) heroData.filtered = true
      filtered.push(heroData)
    }
  }

  return filtered.sort((a, b) => computeSynergy(b) - computeSynergy(a)) // descending
})

function computeSynergy(candidate) {
  let score = 0
  for (const hero of team.value) {
    const setA = new Set(hero.localized_name.toLowerCase())
    const setB = new Set(candidate.localized_name.toLowerCase())
    for (const ch of setA) {
      if (setB.has(ch)) score++
    }
  }
  return score
}

function addHero(hero) {
  if (team.value.length >= 5) return
  team.value.push(hero)
}

function removeHero(hero) {
  team.value = team.value.filter((h) => h.id !== hero.id)
}
</script>
