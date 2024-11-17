<template>
  <v-container>
    <v-row>
      <!-- Header Section -->
      <v-col cols="12" class="text-center my-8">
        <h1 class="text-h3 mb-4">DataDee Marketplace</h1>
        <p class="text-subtitle-1">Discover and integrate AI models for your needs</p>
      </v-col>

      <!-- Search and Filter Section -->
      <v-col cols="12" class="mb-6">
        <v-row>
          <v-col cols="12" sm="8">
            <v-text-field
              v-model="searchQuery"
              label="Search models"
              prepend-inner-icon="mdi-magnify"
              variant="outlined"
              hide-details
            ></v-text-field>
          </v-col>
          <v-col cols="12" sm="4">
            <v-select
              v-model="selectedCategory"
              :items="categories"
              label="Category"
              variant="outlined"
              hide-details
            ></v-select>
          </v-col>
        </v-row>
      </v-col>

      <!-- Models Grid -->
      <v-col v-for="model in models" :key="model.id" cols="12" sm="6" md="4" lg="3">
        <v-card class="h-100">
          <v-img
            :src="model.image"
            height="200"
            cover
          ></v-img>
          <v-card-title>{{ model.name }}</v-card-title>
          <v-card-subtitle>{{ model.category }}</v-card-subtitle>
          <v-card-text>
            <p>{{ model.description }}</p>
            <div class="mt-2">
              <v-chip size="small" color="primary" class="mr-2">
                {{ model.price }}
              </v-chip>
              <v-chip size="small" :color="model.rating >= 4 ? 'success' : 'warning'">
                {{ model.rating }} ★
              </v-chip>
            </div>
          </v-card-text>
          <v-card-actions>
            <v-btn variant="tonal" block>Learn More</v-btn>
          </v-card-actions>
        </v-card>
      </v-col>
    </v-row>
  </v-container>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const searchQuery = ref('')
const selectedCategory = ref('')

const categories = [
  'All Categories',
  'Computer Vision',
  'Natural Language Processing',
  'Time Series',
  'Recommendation Systems',
  '
</script>
