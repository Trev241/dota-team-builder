<template>
  <div class="max-w-6xl mx-auto mt-12">
    <h1 class="text-4xl mb-2 font-bold">Another Dota Synergy Predictor</h1>
    <p class="text-xl mb-6">To be used for fun</p>

    <p class="mb-4">
      If you're here, then I assume you must already know what Dota 2 is. For the uninformed, it's a
      simple game where two teams, each having five players, compete to destroy each other's
      "Ancient" (that's where the game gets its name from -
      <i>Defence of the Ancients</i>). Each player can choose a specific hero to play. To keep it
      very simple, it's like choosing a type of car to enter a race with. You can either choose a
      very lightweight car to speed across, or a monster truck to crush everyone's car in your way.
    </p>

    <p class="mb-4">
      The goal is to predict the optimal team composition. A team that ideally consists of heroes
      forming the perfect synergy.
    </p>

    <h2 class="text-2xl font-bold mb-2">How does it work?</h2>
    <p class="mb-4">
      The neural network used here was trained on
      <a
        target="_blank"
        class="text-blue-600 hover:text-blue-400"
        href="https://archive.ics.uci.edu/dataset/367/dota2+games+results"
        >this dataset</a
      >
      containing about 90,000 matches. Each match contains information about the outcome of the
      match and the heroes that were played in it.
    </p>

    <p class="mb-4">
      Every team has five heroes. For every match in the dataset, four out of five of the winning
      team's heroes were given to the model as input. It was the model's task to predict the fifth
      hero in the team from the initial four that were given. A task of this kind helps the model to
      create embeddings for each hero allowing it to capture latent relationships (such as
      synergies) between different heroes in the game.
    </p>

    <p class="mb-4">
      Given how the model was trained, it will not be able to predict a team composition if less
      than four heroes are given as input. To overcome this, heroes for the second, third and fourth
      slots are recommended by measuring the cosine-similarity of different heroes in their
      embedding space.
    </p>

    <h2 class="text-2xl font-bold mb-2">Some heroes are missing</h2>
    <p>
      The
      <a
        target="_blank"
        class="text-blue-600 hover:text-blue-400"
        href="https://archive.ics.uci.edu/dataset/367/dota2+games+results"
        >dataset</a
      >
      used is from 2016. That's the reason why some heroes are missing. This was not created to be
      taken seriously which is why no effort has been made towards ensuring its modern relevance.
    </p>
  </div>
</template>
