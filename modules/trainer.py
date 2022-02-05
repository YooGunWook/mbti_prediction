from sklearn.metrics import accuracy_score
import torch
import time


class Trainer(object):
    def __init__(
        self,
        model,
        device,
        loss_fn,
        optimizer,
        scheduler,
        warmup_scheduler,
        threshold,
        early_stop=0,
    ):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.threshold = threshold
        self.early_stop = early_stop
        self.trigger = False
        self.early_count = 0
        self.loss_chk = 1e9
        self.acc_chk = -1e9

    def train(self, train_dataloader, valid_dataloader, epoch_index):

        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_count = 0, 0, 0
        self.model.train()

        for step, batch in enumerate(train_dataloader):
            batch_count += 1
            b_input_id = batch[0].to(self.device)
            b_attn_mask = batch[1].to(self.device)
            b_label = batch[2].to(self.device)
            self.model.zero_grad()

            outputs = self.model(sent=b_input_id, attention_mask=b_attn_mask)
            loss = self.loss_fn(b_label, outputs)
            batch_loss += loss.item()
            total_loss += loss.item()

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch
                print(
                    f"{epoch_index + 1:^7} | {step:^7} | {batch_loss / batch_count:^12.6f} | {'-':^12.6f} | {'-':^12.6f} | {'training': ^12} | {time_elapsed:^9.2f}"
                )  # epoch batch trainloss validloss validacc type elapsed
                batch_loss, batch_count = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)
        print(
            f"{epoch_index + 1:^7} | {step:^7} | {avg_train_loss:^12.6f} | {'-':^12.6f} | {'-':^12.6f} | {'avg_train': ^12} | {time_elapsed:^9.2f}"
        )

        if valid_dataloader:
            val_loss, val_acc = self.validation(valid_dataloader)
            time_elapsed = time.time() - t0_epoch

            if self.loss_chk > val_loss:
                self.loss_chk = val_loss
                self.acc_chk = val_acc
                torch.save(self.model.state_dict(), "./model_res/model_lowest_loss.pt")
                print(
                    f"{epoch_index + 1:^7} | {step:^7} | {'-':^12.6f} | {val_loss:^12.6f} | {val_acc:^12.6f} | {'val_lowest_loss': ^12} | {time_elapsed:^9.2f}"
                )

            else:
                self.early_count += 1
                if self.acc_chk < val_acc:
                    self.acc_chk = val_acc
                    print(
                        f"{epoch_index + 1:^7} | {step:^7} | {'-':^12.6f} | {val_loss:^12.6f} | {val_acc:^12.6f} | {'val_highest_acc': ^12} | {time_elapsed:^9.2f}"
                    )
                    torch.save(
                        self.model.state_dict(), "./model_res/model_highest_acc.pt"
                    )
                else:
                    print(
                        f"{epoch_index + 1:^7} | {step:^7} | {'-':^12.6f} | {val_loss:^12.6f} | {val_acc:^12.6f} | {'val_early_count': ^12} | {time_elapsed:^9.2f}"
                    )

            if self.early_count == self.early_stop:
                print("Early Stop!!!")
                self.trigger = True

    def validation(self, dataloader):
        self.model.eval()
        val_loss = 0
        val_acc = 0

        for batch in dataloader:
            b_input_id = batch[0].to(self.device)
            b_attn_mask = batch[1].to(self.device)
            b_label = batch[2].to(self.device)

            with torch.no_grad():
                output = self.model(b_input_id, b_attn_mask)
            pred = (output >= self.threshold).float()
            loss = self.loss_fn(output, b_label)
            acc = accuracy_score(b_label.numpy(), pred.numpy())
            val_loss += loss.item()
            val_acc += acc

        val_loss = val_loss / len(dataloader)
        val_acc = val_acc / len(dataloader)

        return val_loss, val_acc
