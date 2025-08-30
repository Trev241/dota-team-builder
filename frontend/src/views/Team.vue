<template>
  <div class="relative min-h-screen mx-auto p-6">
    <!-- Overlay text on large screens -->
    <div
      v-if="!isSmallScreen && typedText"
      class="fixed inset-0 flex justify-center text-white uppercase items-center font-bold text-8xl select-none pointer-events-none drop-shadow-[0_1.2px_1.2px_rgba(0,0,0,0.8)] z-50"
    >
      {{ typedText }}
    </div>

    <div v-if="!isBackendUp" class="flex justify-center items-center">
      <div class="flex items-center border-2 border-cyan-400 rounded-lg p-4 mb-4">
        <Spinner class="mr-4" />
        <div>Connecting to server. Picks will not be optimized until then.</div>
      </div>
    </div>

    <div class="flex flex-col justify-center items-center min-h-[50vh]">
      <div class="w-full text-center">
        <h1 class="text-5xl tracking-widest font-semibold mb-2">
          <span v-if="team.length < 5">YOUR TURN TO PICK</span>
          <span v-else>PREPARE FOR BATTLE</span>
        </h1>
        <p class="text-2xl mb-6">
          <span v-if="team.length < 5">- Choose heroes to build your team -</span>
          <span v-else>- Your team is ready -</span>
        </p>
      </div>

      <div
        class="w-full flex flex-wrap justify-center max-w-6xl gap-2 mx-auto mb-8 bg-slate-900 rounded-full p-2 cursor-default"
      >
        <div
          v-for="index in maxTeamSize"
          :key="index"
          @click="removeHero($event, team[index - 1])"
          class="w-1/3 md:w-1/6 rounded-lg flex flex-col items-center justify-center"
        >
          <template v-if="team[index - 1]">
            <div
              :class="[
                'w-full hover:bg-rose-600 hover:text-white bg-amber-50 text-black rounded-lg cursor-pointer',
                index === team.length && 'team-chosen-icon',
              ]"
            >
              <img
                :src="team[index - 1].icon_url"
                :alt="team[index - 1].localized_name"
                class="w-full rounded-t-md object-contain"
              />
              <div class="my-1 text-sm font-semibold uppercase text-center truncate">
                {{ team[index - 1].localized_name }}
              </div>
            </div>
          </template>
          <template v-else>
            <div class="w-full flex items-center h-20 md:h-34 w-full rounded-lg bg-gray-800">
              <p class="w-full text-center text-lg italic">Empty Slot</p>
            </div>
          </template>
        </div>
      </div>
    </div>

    <div>
      <h2 class="text-3xl tracking-wider font-semibold mb-2">HERO POOL</h2>

      <p v-if="isBackendUp" class="text-gray-400">
        Heroes are automatically sorted from highest to lowest synergy in the team.
      </p>

      <p v-if="!isSmallScreen" class="text-gray-400 mb-6">
        Search for a hero by typing in its name anywhere on the screen.
      </p>
      <input
        v-else
        v-model="searchQuery"
        placeholder="Search heroes"
        type="text"
        class="w-full p-3 mt-2 mb-6 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
      />

      <!-- <p v-else class="text-center mb-4">
        Heroes will be automatically sorted from highest to lowest synergy.
      </p> -->

      <div ref="listContainer" class="flex flex-wrap gap-2 mb-6">
        <div
          v-for="hero in filteredAndSortedHeroes"
          :key="hero.id"
          :class="[
            'w-1/3 md:w-30 hero-icon rounded-lg flex flex-col items-center cursor-pointer transition-shadow relative',
            team.length >= 5 ? 'opacity-50 pointer-events-none' : 'bg-white',
            !hero.filtered && 'opacity-10',
            hero.selected && 'opacity-10',
          ]"
          @click="addHero(hero)"
          @mouseenter="playHoverAnim($event, true)"
          @mouseleave="playHoverAnim($event, false)"
          :aria-disabled="team.length >= 5"
          :title="team.length >= 5 ? 'Team is full' : hero.localized_name"
        >
          <div
            v-if="hero.score"
            :class="[
              'absolute inset-0 flex justify-end items-end opacity-0 hover:opacity-100 font-bold',
            ]"
          >
            <div class="bg-black p-1 rounded-tl-lg">
              {{ Math.round(hero.score * 100) / 100 }}
            </div>
          </div>
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

      <p class="text-gray-400 text-center cursor-pointer">
        I can't find a hero,
        <a @click="$router.push('/about')" class="text-white hover:text-gray-300 hover:underline"
          >it's missing</a
        >.
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from "vue"
import gsap from "gsap"
import Flip from "gsap/Flip"
import Spinner from "../components/Spinner.vue"

const API_URL = import.meta.env.VITE_API_URL

gsap.registerPlugin(Flip)

const maxTeamSize = 5

const searchQuery = ref("")
const typedText = ref("")
const team = ref([])
const heroes = ref([]) // initially empty, to be fetched

const debounceTimeout = ref(null)
const isSmallScreen = ref(window.innerWidth < 768)
const isBackendUp = ref(false)
const pingInterval = ref(null)

const listContainer = ref(null)
const animationDone = ref(false)

const resetTypedTextDebounced = () => {
  if (debounceTimeout.value) clearTimeout(debounceTimeout.value)
  debounceTimeout.value = setTimeout(() => {
    typedText.value = ""
    searchQuery.value = typedText.value
  }, 3500)
}

const onGlobalKeyDown = async (event) => {
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

const onResize = () => {
  isSmallScreen.value = window.innerWidth < 768
  if (isSmallScreen.value) typedText.value = ""
}

const playHoverAnim = (event, hover) => {
  const element = event.currentTarget
  if (hover) {
    gsap.to(element, { scale: 1.1, duration: 0.2 })
  } else {
    gsap.to(element, { scale: 1, duration: 0.2 })
  }
}

const pingBackend = async () => {
  try {
    console.log(API_URL)
    const response = await fetch(`${API_URL}/health`)
    if (response.ok) {
      isBackendUp.value = true
      clearInterval(pingInterval.value)
      console.log("Backend is up!")
    } else {
      console.log("Backend not ready yet...")
    }
  } catch (error) {
    console.log("Backend not reachable:", error)
  }
}

onMounted(async () => {
  pingInterval.value = setInterval(pingBackend, 10000)

  window.addEventListener("keydown", onGlobalKeyDown)
  window.addEventListener("resize", onResize)

  const response = await fetch("/heroes.json")
  heroes.value = await response.json()

  await nextTick()

  const images = document.querySelectorAll(".hero-icon img")

  gsap.set(".hero-icon", { scale: 0 })
  const imagePromises = Array.from(images).map((img) => {
    return new Promise((resolve) => {
      if (img.complete) {
        resolve()
      } else {
        img.addEventListener("load", resolve)
        img.addEventListener("error", resolve)
      }
    })
  })

  // Wait for all images to load
  await Promise.all(imagePromises)

  await gsap.to(".hero-icon", { scale: 1, stagger: 0.01 })
  animationDone.value = true
})

onUnmounted(() => {
  window.removeEventListener("keydown", onGlobalKeyDown)
  window.removeEventListener("resize", onResize)
  if (debounceTimeout.value) clearTimeout(debounceTimeout.value)
})

const availableHeroes = computed(() =>
  heroes.value.map((h) => ({
    ...h,
    selected: team.value.some((th) => th.id === h.id),
  })),
)

async function animateSort(newOrder) {
  const state = Flip.getState(listContainer.value.querySelectorAll(".hero-icon"))

  // Update with new order
  heroes.value = newOrder

  await nextTick()

  Flip.from(state, {
    duration: 1.25,
    ease: "power3.out",
  })
}

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
    if (a.selected && !b.selected) return 1
    if (!a.selected && b.selected) return -1

    if (a.score !== b.score) return b.score - a.score

    return a.localized_name.localeCompare(b.localized_name)
  })
}

const filteredAndSortedHeroes = computed(() => filterHeroIcons())

const addHero = async (hero) => {
  if (team.value.length >= 5 || team.value.some((h) => h.id === hero.id) || !animationDone.value)
    return

  team.value.push(hero)
  await nextTick()
  await gsap.set(".team-chosen-icon", { position: "relative", zIndex: 50 })
  await gsap.fromTo(
    ".team-chosen-icon",
    { scale: 3 },
    { scale: 1, duration: 0.5, ease: "power4.in" },
  )

  // Do not call predict if the team is full or if backend is not up
  if (team.value.length === 5 || !isBackendUp) {
    animateSort(heroes.value)
    return
  }

  await predictHero()
}

const removeHero = async (event, hero) => {
  if (!hero) return

  const target = event.currentTarget
  target.classList.add("team-removed-icon")

  await gsap.to(target, {
    yPercent: 100,
    opacity: 0,
    ease: "power4.out",
    duration: 0.5,
  })

  gsap.set(target, { yPercent: 0, opacity: 1, position: "relative", zIndex: 0 })
  target.classList.remove("team-removed-icon")

  team.value = team.value.filter((h) => h.id !== hero.id)

  if (team.value.length >= 1) await predictHero()
}

const predictHero = async () => {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ heroes: team.value.map((hero) => hero.id) }),
    })

    if (!response.ok) throw new Error("Prediction API error")

    const results = await response.json()

    heroes.value = results.sorted_candidates || []
    if (!isSmallScreen.value) animateSort(results.sorted_candidates)
  } catch (error) {
    console.error(error)
  }
}
</script>
