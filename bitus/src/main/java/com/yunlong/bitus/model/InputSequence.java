package com.yunlong.bitus.model;

import java.util.List;

public class InputSequence {
    private List<EventStep> sequence;

    public static class EventStep {
        private int eventType; // 0=view, 1=addtocart, 2=transaction
        private float timeDelta; // difference in timestamp (could be normalized)
        private int itemId; // must match item2id mapping

        // Getters and Setters
        public int getEventType() {
            return eventType;
        }

        public void setEventType(int eventType) {
            this.eventType = eventType;
        }

        public float getTimeDelta() {
            return timeDelta;
        }

        public void setTimeDelta(float timeDelta) {
            this.timeDelta = timeDelta;
        }

        public int getItemId() {
            return itemId;
        }

        public void setItemId(int itemId) {
            this.itemId = itemId;
        }
    }

    public List<EventStep> getSequence() {
        return sequence;
    }

    public void setSequence(List<EventStep> sequence) {
        this.sequence = sequence;
    }
}
